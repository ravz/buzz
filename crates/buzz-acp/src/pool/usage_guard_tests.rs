use super::*;
use crate::queue::{BatchEvent, EventQueue, QueuedEvent};
use crate::usage_guard::Settings;
use nostr::{Event, EventBuilder, Keys, Kind};
use serde_json::{json, Value};

const ADAPTER: &str = r#"import json, sys
sessions = {}
for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    params = request["params"]
    if method == "initialize":
        result = {"protocolVersion": 1, "agentCapabilities": {"sessionCapabilities": {"close": {}}}}
    elif method == "session/new":
        session_id = f"session-{len(sessions) + 1}"
        sessions[session_id] = [0, 0, 0]
        result = {"sessionId": session_id}
    elif method == "session/close":
        result = {}
    elif method == "session/prompt":
        text = params["prompt"][0]["text"]
        sid = params["sessionId"]
        usage = sessions[sid]
        usage[0] += 100 if "expensive" in text else 99
        usage[1] += 5
        usage[2] += 31
        print(json.dumps({"jsonrpc": "2.0", "method": "_goose/unstable/session/update",
            "params": {"sessionId": sid, "update": {"sessionUpdate": "usage_update",
                "used": sum(usage[:2]), "contextLimit": 1000, "accumulatedInputTokens": usage[0],
                "accumulatedOutputTokens": usage[1], "accumulatedCachedInputTokens": usage[2]}}}), flush=True)
        result = {"stopReason": "end_turn"}
    else:
        raise ValueError(method)
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"#;

async fn agent(name: &str) -> (OwnedAgent, observer::ObserverHandle) {
    let observer = observer::ObserverHandle::in_process();
    let python = if cfg!(windows) { "python" } else { "python3" };
    let mut acp = AcpClient::spawn(
        python,
        &["-u".into(), "-c".into(), ADAPTER.into()],
        &[],
        false,
    )
    .await
    .unwrap();
    acp.set_observer(Some(observer.clone()), 0);
    timeout(Duration::from_secs(10), acp.initialize())
        .await
        .unwrap()
        .unwrap();
    (
        OwnedAgent {
            index: 0,
            acp,
            state: SessionState::default(),
            model_capabilities: None,
            desired_model: None,
            model_overridden: false,
            desired_model_request_id: None,
            desired_model_pending_ack: false,
            startup_effort: None,
            agent_name: name.into(),
            goose_system_prompt_supported: None,
            protocol_version: 1,
        },
        observer,
    )
}

async fn run_turn(
    agent: OwnedAgent,
    ctx: Arc<PromptContext>,
    batch: Option<FlushBatch>,
    text: &str,
) -> PromptResult {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let text = batch.is_none().then(|| text.to_owned());
    timeout(
        Duration::from_secs(12),
        run_prompt_task(
            agent,
            batch,
            text,
            ctx,
            tx,
            None,
            "guard-test".into(),
            Default::default(),
        ),
    )
    .await
    .expect("prompt fixture should finish");
    rx.recv().await.unwrap()
}

fn wire(observer: &observer::ObserverHandle, method: &str) -> Vec<Value> {
    observer
        .snapshot()
        .into_iter()
        .filter(|event| event.kind == "acp_write" && event.payload["method"] == method)
        .map(|event| event.payload["params"].clone())
        .collect()
}

fn event(content: &str) -> Event {
    EventBuilder::new(Kind::Custom(9), content)
        .sign_with_keys(&Keys::generate())
        .unwrap()
}

fn batch(scope: &SessionScope, content: &str, scheduled_workflow: bool) -> FlushBatch {
    FlushBatch {
        channel_id: scope.channel_id(),
        scope: scope.clone(),
        events: vec![BatchEvent {
            event: event(content),
            prompt_tag: "test".into(),
            received_at: std::time::Instant::now(),
            answer_required: None,
            scheduled_workflow,
            edit: None,
        }],
        cancelled_events: vec![],
        cancel_reason: None,
    }
}

async fn relay_context(settings: Settings) -> (PromptContext, JoinHandle<()>) {
    let (channel_info, _, relay) = tests::counting_resolver(json!([])).await;
    let mut ctx = tests::make_prompt_context_no_owner();
    ctx.rest_client = channel_info.rest_client.clone();
    ctx.channel_info = channel_info;
    ctx.usage_guard = UsageGuard::new(settings);
    (ctx, relay)
}

#[tokio::test]
async fn usage_guard_rotates_at_the_input_threshold_and_ledger_sequences_reset() {
    let dir = tempfile::tempdir().unwrap();
    let mut ctx = tests::make_prompt_context_no_owner();
    ctx.usage_guard.settings.model.rotate = 100;
    ctx.usage_guard.settings.ledger_dir = Some(dir.path().join("ledger"));
    let ctx = Arc::new(ctx);
    let (agent, observer) = agent("adapter").await;
    let first = run_turn(agent, Arc::clone(&ctx), None, "small").await;
    assert!(matches!(
        first.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert_eq!(
        first.agent.state.heartbeat_session.as_deref(),
        Some("session-1")
    );
    let second = run_turn(first.agent, Arc::clone(&ctx), None, "expensive").await;
    assert!(second.agent.state.heartbeat_session.is_none());
    assert!(second.agent.state.pending_close.contains("session-1"));
    let mut third = run_turn(second.agent, Arc::clone(&ctx), None, "small").await;
    assert_eq!(
        third.agent.state.heartbeat_session.as_deref(),
        Some("session-2")
    );
    assert_eq!(
        wire(&observer, "session/close"),
        [json!({"sessionId": "session-1"})]
    );

    let pubkey = ctx.agent_keys.public_key().to_hex();
    let filename = format!("usage-{pubkey}-{}.jsonl", std::process::id());
    let ledger = std::fs::read_to_string(dir.path().join("ledger").join(filename)).unwrap();
    let rows: Vec<Value> = ledger
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let seqs: Vec<_> = rows
        .iter()
        .map(|row| row["turn_seq"].as_u64().unwrap())
        .collect();
    let inputs: Vec<_> = rows
        .iter()
        .map(|row| row["input_tokens"].as_u64().unwrap())
        .collect();
    assert_eq!(seqs, [1, 2, 1]);
    assert_eq!(inputs, [99, 100, 99]);
    assert_eq!(rows[1]["cache_read_tokens"], 31);
    assert_eq!(rows[1]["total_tokens"], 105);
    assert_eq!(rows[2]["session_id"], "session-2");
    third.agent.acp.shutdown().await;
}

#[tokio::test]
async fn usage_guard_rotates_only_enabled_workflow_only_batches() {
    let scope = SessionScope::Thread {
        channel_id: Uuid::new_v4(),
        root_event_id: "a".repeat(64),
    };
    let (agent, _) = agent("adapter").await;
    let (off, off_relay) = relay_context(Settings::default()).await;
    let first = run_turn(
        agent,
        Arc::new(off),
        Some(batch(&scope, "scheduled", true)),
        "",
    )
    .await;
    assert_eq!(
        first.agent.state.scope_session_id(&scope),
        Some("session-1")
    );
    let mut on = Settings::default();
    on.model.rotate_scheduled = true;
    let (on, on_relay) = relay_context(on).await;
    let on = Arc::new(on);
    let mut mixed = batch(&scope, "scheduled", true);
    mixed.cancelled_events = batch(&scope, "human", false).events;
    let mixed = run_turn(first.agent, Arc::clone(&on), Some(mixed), "").await;
    assert_eq!(
        mixed.agent.state.scope_session_id(&scope),
        Some("session-1")
    );
    let mut last = run_turn(mixed.agent, on, Some(batch(&scope, "scheduled", true)), "").await;
    assert!(last.agent.state.scope_session_id(&scope).is_none());
    assert!(last.agent.state.pending_close.contains("session-1"));
    last.agent.acp.shutdown().await;
    off_relay.abort();
    on_relay.abort();
}

#[tokio::test]
async fn usage_guard_rotates_after_expensive_setup_without_repeating_setup() {
    let scope = SessionScope::Conversation {
        channel_id: Uuid::new_v4(),
    };
    let mut settings = Settings::default();
    settings.model.rotate = 100;
    let (mut ctx, relay) = relay_context(settings).await;
    ctx.initial_message = Some("expensive setup".into());
    let (agent, observer) = agent("adapter").await;

    let mut result = run_turn(
        agent,
        Arc::new(ctx),
        Some(batch(&scope, "small", false)),
        "",
    )
    .await;

    assert!(matches!(
        result.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert_eq!(
        result.agent.state.scope_session_id(&scope),
        Some("session-2")
    );
    assert_eq!(
        wire(&observer, "session/close"),
        [json!({"sessionId": "session-1"})]
    );
    let prompts = wire(&observer, "session/prompt");
    assert_eq!(prompts.len(), 2);
    assert_eq!(prompts[0]["sessionId"], "session-1");
    assert_eq!(prompts[1]["sessionId"], "session-2");
    result.agent.acp.shutdown().await;
    relay.abort();
}

#[tokio::test]
async fn usage_guard_compacts_claude_in_session_and_applies_inner_turn_metadata_only_to_claude() {
    for (name, claude) in [("claude-agent-acp", true), ("other", false)] {
        let (agent, observer) = agent(name).await;
        let mut ctx = tests::make_prompt_context_no_owner();
        ctx.usage_guard.settings.claude.compact = 100;
        ctx.usage_guard.settings.claude_max_inner_turns = 7;
        let ctx = Arc::new(ctx);
        let first = run_turn(agent, Arc::clone(&ctx), None, "expensive").await;
        let mut second = run_turn(first.agent, ctx, None, "small").await;
        let session = wire(&observer, "session/new");
        let metadata = session[0]
            .pointer("/_meta/claudeCode/options/maxTurns")
            .cloned();
        assert_eq!(metadata, claude.then(|| json!(7)));
        let prompts = wire(&observer, "session/prompt");
        assert_eq!(prompts.len(), if claude { 3 } else { 2 });
        if claude {
            assert_eq!(prompts[1]["prompt"][0]["text"], "/compact");
            assert_eq!(prompts[1]["sessionId"], "session-1");
        }
        assert_eq!(
            second.agent.state.heartbeat_session.as_deref(),
            Some("session-1")
        );
        second.agent.acp.shutdown().await;
    }
}

#[tokio::test]
async fn usage_guard_admission_keeps_events_in_the_queue_until_reopened_in_drop_mode() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gate");
    let mut settings = Settings::default();
    settings.model.admission_file = Some(path.clone());
    let (ctx, relay) = relay_context(settings).await;
    let ctx = Arc::new(ctx);
    let (agent, observer) = agent("adapter").await;
    let mut pool = AgentPool::from_slots(vec![Some(agent)]);
    let scope = SessionScope::Conversation {
        channel_id: Uuid::new_v4(),
    };
    let mut queue = EventQueue::new(DedupMode::Drop);
    assert!(queue.push(QueuedEvent {
        channel_id: scope.channel_id(),
        scope: scope.clone(),
        event: event("pending"),
        received_at: std::time::Instant::now(),
        prompt_tag: "test".into(),
        answer_required: None,
        scheduled_workflow: false,
        edit: None,
    }));
    let mut last_activity = tokio::time::Instant::now();
    assert!(
        crate::dispatch_pending(&mut pool, &mut queue, &ctx, &mut last_activity, None).is_empty()
    );
    assert_eq!(queue.queued_event_count(scope.channel_id()), 1);
    assert!(wire(&observer, "session/prompt").is_empty());
    std::fs::write(path, "open\n").unwrap();
    assert_eq!(
        crate::dispatch_pending(&mut pool, &mut queue, &ctx, &mut last_activity, None).len(),
        1
    );
    let mut result = timeout(Duration::from_secs(10), pool.result_rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        result.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert_eq!(queue.queued_event_count(scope.channel_id()), 0);
    assert_eq!(wire(&observer, "session/prompt").len(), 1);
    result.agent.acp.shutdown().await;
    relay.abort();
}
