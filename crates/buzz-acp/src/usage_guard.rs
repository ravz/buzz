//! Operator-configured admission, context reduction, and local usage accounting.

use std::{
    collections::VecDeque,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{Duration, Instant},
};

use clap::{parser::ValueSource, ArgMatches, Args, Command, FromArgMatches};
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{config::ConfigError, usage::TurnUsage};

const DEFAULT_WINDOW_SECS: u64 = 600;

#[derive(Debug, Args)]
struct UsageGuardFlags {
    /// Inclusive input tokens in a completed ACP turn that trigger context reduction before the next prompt. Claude uses `/compact`; adapters without a portable compact command rotate the session instead.
    #[arg(
        long,
        env = "BUZZ_ACP_COMPACT_INPUT_TOKENS",
        default_value_t = 0,
        hide_env_values = true
    )]
    compact_input_tokens: u64,
    /// Inclusive input tokens in a completed ACP turn that rotate the session.
    #[arg(
        long,
        env = "BUZZ_ACP_ROTATE_INPUT_TOKENS",
        default_value_t = 0,
        hide_env_values = true
    )]
    rotate_input_tokens: u64,
    /// Per-process rolling input/cache allowance for every adapter. Crossing it closes the configured admission gate. 0 disables the breaker.
    #[arg(
        long,
        env = "BUZZ_ACP_AGENT_WINDOW_TOKENS",
        default_value_t = 0,
        hide_env_values = true
    )]
    agent_window_tokens: u64,
    /// Rolling window used by the provider-neutral circuit breaker.
    #[arg(long, env = "BUZZ_ACP_USAGE_WINDOW_SECS", default_value_t = DEFAULT_WINDOW_SECS)]
    usage_window_secs: u64,
    /// Scope-local fail-closed gate for every model. Exact content `open` admits work; missing, unreadable, or any other content blocks turns.
    #[arg(long, env = "BUZZ_ACP_ADMISSION_FILE")]
    admission_file: Option<PathBuf>,
    /// Write provider-neutral turn usage JSONL files for the host aggregate monitor. One file is used per agent identity and harness process.
    #[arg(long, env = "BUZZ_ACP_USAGE_LEDGER_DIR")]
    usage_ledger_dir: Option<PathBuf>,
    /// Rotate all model sessions after non-answer-required workflow turns.
    #[arg(long, env = "BUZZ_ACP_ROTATE_SCHEDULED_TURNS", default_value_t = false, action = clap::ArgAction::Set)]
    rotate_scheduled_turns: bool,
    /// Claude-only outer ACP prompts allowed before session rotation. 0 falls back to `max_turns_per_session`.
    #[arg(
        long,
        env = "BUZZ_ACP_CLAUDE_MAX_TURNS_PER_SESSION",
        default_value_t = 0
    )]
    claude_max_turns_per_session: u32,
    /// Claude-only wall-clock cap for one ACP prompt. 0 falls back to the generic maximum turn duration.
    #[arg(long, env = "BUZZ_ACP_CLAUDE_MAX_TURN_DURATION", default_value_t = 0)]
    claude_max_turn_duration_secs: u64,
    /// Maximum Claude SDK agentic turns inside one ACP prompt. 0 leaves the adapter default unchanged. Passed out-of-band in session/new metadata.
    #[arg(long, env = "BUZZ_ACP_CLAUDE_MAX_INNER_TURNS", default_value_t = 0)]
    claude_max_inner_turns: u64,
    /// Inclusive Claude input tokens in a completed ACP turn that schedule a `/compact` before the next prompt. 0 disables proactive compaction.
    #[arg(
        long,
        env = "BUZZ_ACP_CLAUDE_COMPACT_INPUT_TOKENS",
        default_value_t = 0,
        hide_env_values = true
    )]
    claude_compact_input_tokens: u64,
    /// Inclusive Claude input tokens in a completed ACP turn that rotate the session before any later prompt. 0 disables usage-based rotation.
    #[arg(
        long,
        env = "BUZZ_ACP_CLAUDE_ROTATE_INPUT_TOKENS",
        default_value_t = 0,
        hide_env_values = true
    )]
    claude_rotate_input_tokens: u64,
    /// Per-process rolling Claude input/cache allowance. Crossing it closes the configured admission gate. 0 disables the circuit breaker.
    #[arg(
        long,
        env = "BUZZ_ACP_CLAUDE_AGENT_WINDOW_TOKENS",
        default_value_t = 0,
        hide_env_values = true
    )]
    claude_agent_window_tokens: u64,
    /// Rolling window used by the per-process Claude circuit breaker.
    #[arg(long, env = "BUZZ_ACP_CLAUDE_USAGE_WINDOW_SECS", default_value_t = DEFAULT_WINDOW_SECS)]
    claude_usage_window_secs: u64,
    /// Shared text file controlling new Claude turns. The exact content `open` admits work; missing, unreadable, or any other content fails closed. Non-Claude adapters ignore this file.
    #[arg(long, env = "BUZZ_ACP_CLAUDE_ADMISSION_FILE")]
    claude_admission_file: Option<PathBuf>,
    /// Rotate Claude channel sessions after non-answer-required workflow turns.
    #[arg(long, env = "BUZZ_ACP_CLAUDE_ROTATE_SCHEDULED_TURNS", default_value_t = false, action = clap::ArgAction::Set)]
    claude_rotate_scheduled_turns: bool,
    /// Deterministic per-identity startup delay spread over 0..N seconds.
    #[arg(long, env = "BUZZ_ACP_STARTUP_JITTER_SECS", default_value_t = 0)]
    startup_jitter_secs: u64,
}

#[derive(Clone, Debug)]
pub(crate) struct Policy {
    pub(crate) compact: u64,
    pub(crate) rotate: u64,
    pub(crate) rotate_scheduled: bool,
    pub(crate) window_tokens: u64,
    pub(crate) window_secs: u64,
    pub(crate) admission_file: Option<PathBuf>,
}

impl Default for Policy {
    fn default() -> Self {
        Self {
            compact: 0,
            rotate: 0,
            rotate_scheduled: false,
            window_tokens: 0,
            window_secs: DEFAULT_WINDOW_SECS,
            admission_file: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Settings {
    pub(crate) model: Policy,
    pub(crate) claude: Policy,
    pub(crate) claude_max_turns: u32,
    pub(crate) claude_max_duration_secs: u64,
    pub(crate) claude_max_inner_turns: u64,
    pub(crate) ledger_dir: Option<PathBuf>,
    pub(crate) startup_jitter_secs: u64,
}

// Keep the legacy help defaults while allowing a CLI/env value of 0 or false
// to override a nonzero/true generic setting for Claude.
impl FromArgMatches for Settings {
    fn from_arg_matches(matches: &ArgMatches) -> Result<Self, clap::Error> {
        let flags = UsageGuardFlags::from_arg_matches(matches)?;
        let supplied = |id| matches.value_source(id) != Some(ValueSource::DefaultValue);
        let override_number = |id, value, generic| if supplied(id) { value } else { generic };
        let model = Policy {
            compact: flags.compact_input_tokens,
            rotate: flags.rotate_input_tokens,
            rotate_scheduled: flags.rotate_scheduled_turns,
            window_tokens: flags.agent_window_tokens,
            window_secs: flags.usage_window_secs,
            admission_file: flags.admission_file,
        };
        let claude = Policy {
            compact: override_number(
                "claude_compact_input_tokens",
                flags.claude_compact_input_tokens,
                model.compact,
            ),
            rotate: override_number(
                "claude_rotate_input_tokens",
                flags.claude_rotate_input_tokens,
                model.rotate,
            ),
            rotate_scheduled: if supplied("claude_rotate_scheduled_turns") {
                flags.claude_rotate_scheduled_turns
            } else {
                model.rotate_scheduled
            },
            window_tokens: override_number(
                "claude_agent_window_tokens",
                flags.claude_agent_window_tokens,
                model.window_tokens,
            ),
            window_secs: override_number(
                "claude_usage_window_secs",
                flags.claude_usage_window_secs,
                model.window_secs,
            ),
            admission_file: flags
                .claude_admission_file
                .or_else(|| model.admission_file.clone()),
        };
        Ok(Self {
            model,
            claude,
            claude_max_turns: flags.claude_max_turns_per_session,
            claude_max_duration_secs: flags.claude_max_turn_duration_secs,
            claude_max_inner_turns: flags.claude_max_inner_turns,
            ledger_dir: flags.usage_ledger_dir,
            startup_jitter_secs: flags.startup_jitter_secs,
        })
    }

    fn update_from_arg_matches(&mut self, matches: &ArgMatches) -> Result<(), clap::Error> {
        *self = Self::from_arg_matches(matches)?;
        Ok(())
    }
}

impl Args for Settings {
    fn augment_args(cmd: Command) -> Command {
        UsageGuardFlags::augment_args(cmd)
    }

    fn augment_args_for_update(cmd: Command) -> Command {
        Self::augment_args(cmd)
    }
}

impl Settings {
    pub(crate) fn validate(&self) -> Result<(), ConfigError> {
        for (prefix, policy) in [("", &self.model), ("claude_", &self.claude)] {
            if policy.compact != 0 && policy.rotate != 0 && policy.compact >= policy.rotate {
                return Err(ConfigError::ConfigFile(format!(
                    "{prefix}compact_input_tokens ({}) must be lower than {prefix}rotate_input_tokens ({})",
                    policy.compact, policy.rotate,
                )));
            }
        }
        if self.claude_max_duration_secs > crate::config::MAX_TURN_DURATION_CEILING_SECS {
            return Err(ConfigError::ConfigFile(format!(
                "claude_max_turn_duration_secs ({}) exceeds ceiling ({}s / 7 days)",
                self.claude_max_duration_secs,
                crate::config::MAX_TURN_DURATION_CEILING_SECS,
            )));
        }
        Ok(())
    }

    pub(crate) fn policy(&self, claude: bool) -> &Policy {
        if claude {
            &self.claude
        } else {
            &self.model
        }
    }

    pub(crate) fn has_admission_file(&self) -> bool {
        self.model.admission_file.is_some() || self.claude.admission_file.is_some()
    }

    pub(crate) fn max_duration(&self, claude: bool, generic: Duration) -> Duration {
        if claude && self.claude_max_duration_secs != 0 {
            Duration::from_secs(self.claude_max_duration_secs)
        } else {
            generic
        }
    }

    pub(crate) fn max_turns(&self, claude: bool, generic: u32) -> u32 {
        if claude && self.claude_max_turns != 0 {
            self.claude_max_turns
        } else {
            generic
        }
    }

    pub(crate) fn startup_delay(&self, pubkey: &nostr::PublicKey) -> Duration {
        if self.startup_jitter_secs == 0 {
            return Duration::ZERO;
        }
        let hash = Sha256::digest(pubkey.as_bytes());
        let mut prefix = [0; 8];
        prefix.copy_from_slice(&hash[..8]);
        Duration::from_secs(u64::from_be_bytes(prefix) % self.startup_jitter_secs)
    }

    pub(crate) fn summary(&self) -> String {
        format!(
            "model_compact={} model_rotate={} model_rotate_scheduled={} model_window={} claude_compact={} claude_rotate={} claude_rotate_scheduled={} claude_window={} usage_ledger={} startup_jitter={}",
            self.model.compact, self.model.rotate, self.model.rotate_scheduled, self.model.window_tokens,
            self.claude.compact, self.claude.rotate, self.claude.rotate_scheduled, self.claude.window_tokens,
            self.ledger_dir.as_ref().map_or_else(|| "off".into(), |path| path.display().to_string()),
            self.startup_jitter_secs,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Reduction {
    #[default]
    None,
    Compact,
    Rotate,
}

impl Policy {
    pub(crate) fn reduction(&self, claude: bool, scheduled: bool, input: u64) -> Reduction {
        if (scheduled && self.rotate_scheduled) || (self.rotate != 0 && input >= self.rotate) {
            Reduction::Rotate
        } else if self.compact != 0 && input >= self.compact {
            if claude {
                Reduction::Compact
            } else {
                Reduction::Rotate
            }
        } else {
            Reduction::None
        }
    }
}

fn admission_file_is_open(path: &Path) -> bool {
    if !path.metadata().is_ok_and(|meta| meta.is_file()) {
        return false;
    }
    let mut content = String::new();
    // A small control file should never allocate unbounded memory.
    File::open(path)
        .and_then(|file| file.take(4097).read_to_string(&mut content))
        .is_ok_and(|size| size <= 4096 && content.trim_end() == "open")
}

#[derive(Default)]
struct Window {
    turns: VecDeque<(Instant, u64)>,
    total: u128,
    close_pending: bool,
}

impl Window {
    fn record(&mut self, now: Instant, tokens: u64, policy: &Policy) -> bool {
        let duration = Duration::from_secs(policy.window_secs);
        while self
            .turns
            .front()
            .is_some_and(|(at, _)| now.duration_since(*at) >= duration)
        {
            if let Some((_, expired)) = self.turns.pop_front() {
                self.total -= u128::from(expired);
            }
        }
        if tokens != 0 {
            self.turns.push_back((now, tokens));
            self.total += u128::from(tokens);
        }
        self.total > u128::from(policy.window_tokens)
    }

    fn close_gate(&mut self, path: &Path) {
        self.close_pending = match fs::write(path, b"closed") {
            Ok(()) => false,
            Err(error) => {
                tracing::warn!(%error, path = %path.display(), "failed to write closed admission gate");
                true
            }
        };
    }
}

/// One runtime per harness process, shared by all workers and their respawns.
#[derive(Default)]
pub(crate) struct UsageGuard {
    pub(crate) settings: Settings,
    window: Mutex<Window>,
    ledger_lock: Mutex<()>,
}

impl UsageGuard {
    pub(crate) fn new(settings: Settings) -> Self {
        Self {
            settings,
            ..Self::default()
        }
    }

    pub(crate) fn admits(&self, claude: bool) -> bool {
        let Some(path) = self.settings.policy(claude).admission_file.as_deref() else {
            return true;
        };
        let Ok(mut window) = self.window.lock() else {
            return false;
        };
        if window.close_pending {
            window.close_gate(path);
            return false;
        }
        admission_file_is_open(path)
    }

    pub(crate) fn record_window(&self, claude: bool, input: u64) {
        let policy = self.settings.policy(claude);
        if let Some(path) = policy
            .admission_file
            .as_deref()
            .filter(|_| policy.window_tokens != 0)
        {
            if let Ok(mut window) = self.window.lock() {
                // TurnUsage's input already includes cache reads and writes.
                if window.record(Instant::now(), input, policy) {
                    window.close_gate(path);
                }
            }
        }
    }

    pub(crate) fn append(&self, row: &LedgerRow<'_>) {
        let Some(dir) = &self.settings.ledger_dir else {
            return;
        };
        let path = dir.join(format!("usage-{}-{}.jsonl", row.pubkey, row.pid));
        let result = (|| -> std::io::Result<()> {
            let _guard = self
                .ledger_lock
                .lock()
                .map_err(|error| std::io::Error::other(error.to_string()))?;
            let mut line = serde_json::to_vec(row)?;
            line.push(b'\n');
            fs::create_dir_all(dir)?;
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)?
                .write_all(&line)
        })();
        if let Err(error) = result {
            tracing::warn!(%error, path = %path.display(), "failed to append ACP usage ledger");
        }
    }
}

#[derive(Serialize)]
pub(crate) struct LedgerRow<'a> {
    pub(crate) admission_file: Option<&'a Path>,
    pub(crate) agent_name: &'a str,
    cache_read_tokens: u64,
    cache_write_tokens: u64,
    cost_usd: Option<f64>,
    pub(crate) harness: &'a str,
    input_tokens: u64,
    model: Option<&'a str>,
    output_tokens: u64,
    pub(crate) pid: u32,
    pub(crate) pubkey: &'a str,
    pub(crate) session_id: &'a str,
    timestamp_utc: String,
    total_tokens: u64,
    pub(crate) turn_seq: u64,
}

impl<'a> LedgerRow<'a> {
    pub(crate) fn new(
        identity: (&'a str, &'a str, &'a str),
        admission_file: Option<&'a Path>,
        session_id: &'a str,
        turn_seq: u64,
        usage: Option<&'a TurnUsage>,
    ) -> Self {
        let input_tokens = usage.and_then(|u| u.turn_input_tokens).unwrap_or(0);
        let output_tokens = usage.and_then(|u| u.turn_output_tokens).unwrap_or(0);
        Self {
            admission_file,
            agent_name: identity.0,
            cache_read_tokens: usage.and_then(|u| u.turn_cache_read_tokens).unwrap_or(0),
            cache_write_tokens: usage.and_then(|u| u.turn_cache_write_tokens).unwrap_or(0),
            cost_usd: usage.and_then(|u| u.turn_cost_usd),
            harness: identity.1,
            input_tokens,
            model: usage.and_then(|u| u.model.as_deref()),
            output_tokens,
            pid: std::process::id(),
            pubkey: identity.2,
            session_id,
            timestamp_utc: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Nanos, false),
            total_tokens: usage
                .and_then(|u| u.turn_total_tokens)
                .unwrap_or_else(|| input_tokens.saturating_add(output_tokens)),
            turn_seq,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::collections::BTreeSet;

    fn args(flags: &[&str]) -> crate::config::CliArgs {
        let key = "1".repeat(64);
        crate::config::CliArgs::try_parse_from(
            ["buzz-acp", "--private-key", &key]
                .into_iter()
                .chain(flags.iter().copied()),
        )
        .unwrap()
    }

    #[test]
    fn flags_default_to_disabled_with_600_second_windows() {
        let settings = args(&[]).usage_guards;
        for policy in [&settings.model, &settings.claude] {
            assert_eq!(
                (
                    policy.compact,
                    policy.rotate,
                    policy.window_tokens,
                    policy.window_secs
                ),
                (0, 0, 0, 600)
            );
            assert!(!policy.rotate_scheduled);
            assert!(policy.admission_file.is_none());
        }
        assert_eq!(
            (
                settings.claude_max_turns,
                settings.claude_max_duration_secs,
                settings.claude_max_inner_turns
            ),
            (0, 0, 0)
        );
        assert_eq!(settings.startup_jitter_secs, 0);
        assert!(settings.ledger_dir.is_none());
    }

    #[test]
    fn flags_parse_and_claude_overrides_only_apply_to_claude() {
        let settings = args(&[
            "--compact-input-tokens",
            "10",
            "--rotate-input-tokens",
            "20",
            "--agent-window-tokens",
            "30",
            "--usage-window-secs",
            "40",
            "--admission-file",
            "model.txt",
            "--usage-ledger-dir",
            "ledger",
            "--rotate-scheduled-turns",
            "true",
            "--claude-compact-input-tokens",
            "0",
            "--claude-rotate-input-tokens",
            "25",
            "--claude-agent-window-tokens",
            "0",
            "--claude-usage-window-secs",
            "600",
            "--claude-admission-file",
            "claude.txt",
            "--claude-rotate-scheduled-turns",
            "false",
            "--claude-max-turns-per-session",
            "3",
            "--claude-max-turn-duration-secs",
            "70",
            "--claude-max-inner-turns",
            "5",
            "--startup-jitter-secs",
            "80",
        ])
        .usage_guards;
        let generic = settings.policy(false);
        assert_eq!(
            (
                generic.compact,
                generic.rotate,
                generic.window_tokens,
                generic.window_secs
            ),
            (10, 20, 30, 40)
        );
        assert!(generic.rotate_scheduled);
        assert_eq!(
            generic.admission_file.as_deref(),
            Some(Path::new("model.txt"))
        );
        let claude = settings.policy(true);
        assert_eq!(
            (
                claude.compact,
                claude.rotate,
                claude.window_tokens,
                claude.window_secs
            ),
            (0, 25, 0, 600)
        );
        assert!(!claude.rotate_scheduled);
        assert_eq!(
            claude.admission_file.as_deref(),
            Some(Path::new("claude.txt"))
        );
        assert_eq!(settings.max_turns(false, 9), 9);
        assert_eq!(settings.max_turns(true, 9), 3);
        assert_eq!(
            settings.max_duration(false, Duration::from_secs(60)),
            Duration::from_secs(60)
        );
        assert_eq!(
            settings.max_duration(true, Duration::from_secs(60)),
            Duration::from_secs(70)
        );
        assert_eq!(settings.claude_max_inner_turns, 5);
        assert_eq!(settings.startup_jitter_secs, 80);
        assert_eq!(settings.ledger_dir.as_deref(), Some(Path::new("ledger")));

        let inherited = args(&[
            "--agent-window-tokens",
            "555",
            "--rotate-scheduled-turns",
            "true",
        ])
        .usage_guards;
        assert_eq!(inherited.policy(true).window_tokens, 555);
        assert!(inherited.policy(true).rotate_scheduled);
    }

    #[test]
    fn startup_rejects_compact_at_or_above_rotation() {
        for (flags, message) in [
            (["--compact-input-tokens", "50", "--rotate-input-tokens", "50"],
             "compact_input_tokens (50) must be lower than rotate_input_tokens (50)"),
            (["--claude-compact-input-tokens", "60", "--claude-rotate-input-tokens", "40"],
             "claude_compact_input_tokens (60) must be lower than claude_rotate_input_tokens (40)"),
        ] {
            let result = crate::config::Config::from_args(args(&flags));
            assert!(matches!(result, Err(ConfigError::ConfigFile(actual)) if actual == message));
        }
        assert!(crate::config::Config::from_args(args(&[
            "--compact-input-tokens",
            "49",
            "--rotate-input-tokens",
            "50",
        ]))
        .is_ok());
    }

    #[test]
    fn admission_gate_accepts_only_open_with_optional_trailing_whitespace() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gate");
        let guard = UsageGuard::new(Settings {
            model: Policy {
                admission_file: Some(path.clone()),
                ..Policy::default()
            },
            ..Settings::default()
        });
        assert!(!guard.admits(false));
        for (content, open) in [
            (&b"open"[..], true),
            (&b"open\r\n "[..], true),
            (&b" open"[..], false),
            (&b"closed"[..], false),
            (&b"OPEN"[..], false),
            (&[0xff][..], false),
        ] {
            fs::write(&path, content).unwrap();
            assert_eq!(guard.admits(false), open);
        }
        fs::remove_file(&path).unwrap();
        fs::create_dir(&path).unwrap();
        assert!(!guard.admits(false));
        assert!(UsageGuard::default().admits(false));
    }

    #[test]
    fn window_breaker_closes_only_when_the_inclusive_input_sum_exceeds_allowance() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gate");
        let policy = Policy {
            window_tokens: 100,
            window_secs: 60,
            admission_file: Some(path.clone()),
            ..Policy::default()
        };
        let guard = UsageGuard::new(Settings {
            model: policy.clone(),
            ..Settings::default()
        });
        fs::write(&path, "open").unwrap();
        guard.record_window(false, 60);
        guard.record_window(false, 40);
        assert!(guard.admits(false));
        guard.record_window(false, 1);
        assert!(!guard.admits(false));
        assert_eq!(fs::read_to_string(path).unwrap(), "closed");

        let now = Instant::now();
        let mut window = Window::default();
        assert!(!window.record(now, 90, &policy));
        assert!(!window.record(now + Duration::from_secs(60), 90, &policy));
        assert!(window.record(now + Duration::from_secs(61), 11, &policy));
    }

    #[test]
    fn reductions_respect_thresholds_adapter_and_scheduled_setting() {
        let mut policy = Policy {
            compact: 100,
            rotate: 200,
            ..Policy::default()
        };
        assert_eq!(policy.reduction(false, false, 99), Reduction::None);
        assert_eq!(policy.reduction(false, false, 100), Reduction::Rotate);
        assert_eq!(policy.reduction(true, false, 100), Reduction::Compact);
        assert_eq!(policy.reduction(true, false, 200), Reduction::Rotate);
        assert_eq!(policy.reduction(false, true, 0), Reduction::None);
        policy.rotate_scheduled = true;
        assert_eq!(policy.reduction(false, true, 0), Reduction::Rotate);
        assert_eq!(policy.reduction(false, false, 0), Reduction::None);
    }

    #[test]
    fn ledger_serializes_the_exact_keys_and_utc_nanoseconds() {
        let row = LedgerRow::new(("adapter", "harness", "pubkey"), None, "session", 1, None);
        let serialized = serde_json::to_string(&row).unwrap();
        let data: serde_json::Value = serde_json::from_str(&serialized).unwrap();
        let keys: BTreeSet<&str> = data
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        assert_eq!(
            keys,
            BTreeSet::from([
                "admission_file",
                "agent_name",
                "cache_read_tokens",
                "cache_write_tokens",
                "cost_usd",
                "harness",
                "input_tokens",
                "model",
                "output_tokens",
                "pid",
                "pubkey",
                "session_id",
                "timestamp_utc",
                "total_tokens",
                "turn_seq",
            ])
        );
        let timestamp = data["timestamp_utc"].as_str().unwrap();
        assert_eq!(timestamp.len(), 35);
        assert_eq!(&timestamp[19..20], ".");
        assert!(timestamp[20..29].bytes().all(|byte| byte.is_ascii_digit()));
        assert_eq!(&timestamp[29..], "+00:00");
        chrono::DateTime::parse_from_rfc3339(timestamp).unwrap();
        assert_eq!(data["turn_seq"], 1);
        assert_eq!(data["pid"], std::process::id());
        assert!(
            data["admission_file"].is_null()
                && data["model"].is_null()
                && data["cost_usd"].is_null()
        );
    }

    #[test]
    fn startup_jitter_is_deterministic_and_bounded_by_pubkey() {
        let first = nostr::Keys::parse(&"1".repeat(64)).unwrap().public_key();
        let second = nostr::Keys::parse(&"2".repeat(64)).unwrap().public_key();
        let settings = Settings {
            startup_jitter_secs: 60_000,
            ..Settings::default()
        };
        assert_eq!(Settings::default().startup_delay(&first), Duration::ZERO);
        assert_eq!(
            settings.startup_delay(&first),
            settings.startup_delay(&first)
        );
        assert_ne!(
            settings.startup_delay(&first),
            settings.startup_delay(&second)
        );
        for key in [first, second] {
            assert!(settings.startup_delay(&key).as_secs() < settings.startup_jitter_secs);
        }
    }
}
