use super::*;
use serde_json::{json, Value};

const ADAPTER: &str = r#"import json, sys
capability = json.loads(sys.argv[1])
next_session = 0
late_close_id = None
for line in sys.stdin:
    request = json.loads(line)
    method = request["method"]
    response = {"jsonrpc": "2.0", "id": request["id"]}
    if method == "initialize":
        session_caps = {} if capability == "absent" else {"close": capability}
        response["result"] = {"protocolVersion": 1,
                              "agentCapabilities": {"sessionCapabilities": session_caps}}
    elif method == "session/new":
        next_session += 1
        response["result"] = {"sessionId": f"session-{next_session}"}
    elif method == "session/close":
        session_id = request["params"]["sessionId"]
        if session_id == "times-out":
            late_close_id = request["id"]
            continue
        if session_id == "rejected":
            response["error"] = {"code": -32603, "message": "close failed"}
        else:
            response["result"] = {}
    elif method == "session/prompt":
        if late_close_id is not None:
            print(json.dumps({"jsonrpc": "2.0", "id": late_close_id, "result": {}}), flush=True)
            late_close_id = None
        response["result"] = {"stopReason": "end_turn"}
    else:
        response["error"] = {"code": -32603, "message": "setup failed"}
    print(json.dumps(response), flush=True)
"#;

fn adapter_config(capability: Value) -> crate::config::Config {
    let mut config = crate::build_mcp_servers_tests::test_config();
    config.agent_command = if cfg!(windows) { "python" } else { "python3" }.into();
    config.agent_args = vec![
        "-u".into(),
        "-c".into(),
        ADAPTER.into(),
        capability.to_string(),
    ];
    config
}

async fn start_agent(
    config: &crate::config::Config,
    observer: &observer::ObserverHandle,
) -> OwnedAgent {
    let result = timeout(
        Duration::from_secs(10),
        crate::spawn_and_init(
            &config.agent_command,
            &config.agent_args,
            &config.persona_env_vars,
            config.has_generated_codex_config,
            0,
            Some(observer.clone()),
        ),
    )
    .await
    .expect("initialize the test adapter within 10s");
    crate::RespawnResult { index: 0, result }
        .into_agent(config)
        .expect("initialize the test adapter")
}

async fn run_heartbeat(agent: OwnedAgent, ctx: PromptContext) -> PromptResult {
    let (result_tx, mut result_rx) = mpsc::unbounded_channel();
    timeout(
        Duration::from_secs(20),
        run_prompt_task(
            agent,
            None,
            Some("heartbeat".into()),
            Arc::new(ctx),
            result_tx,
            None,
            "test-turn".into(),
            Default::default(),
        ),
    )
    .await
    .expect("heartbeat should finish");
    result_rx.recv().await.expect("heartbeat result")
}

fn wire_requests(observer: &observer::ObserverHandle) -> Vec<Value> {
    observer
        .snapshot()
        .into_iter()
        .filter(|event| event.kind == "acp_write")
        .map(|event| event.payload)
        .collect()
}

#[test]
fn replacement_and_bulk_invalidation_deduplicate_abandoned_ids() {
    let mut state = SessionState::default();
    let scope = SessionScope::Conversation {
        channel_id: Uuid::new_v4(),
    };
    state.install_scope_session(scope.clone(), "first".into());
    state.install_scope_session(scope, "second".into());
    assert_eq!(state.pending_close, HashSet::from(["first".into()]));
    state.install_heartbeat_session("second".into());

    state.invalidate_all();

    assert_eq!(
        state.pending_close,
        HashSet::from(["first".into(), "second".into()])
    );
}

#[test]
fn live_ids_are_never_pending_close_across_scopes_or_heartbeat() {
    let mut state = SessionState::default();
    let channel_id = Uuid::new_v4();
    let first = SessionScope::Thread {
        channel_id,
        root_event_id: "a".repeat(64),
    };
    let second = SessionScope::Thread {
        channel_id,
        root_event_id: "b".repeat(64),
    };
    state.install_scope_session(first.clone(), "shared".into());
    state.install_scope_session(second.clone(), "shared".into());
    state.invalidate_scope(&first);
    assert!(state.pending_close.is_empty());

    state.install_heartbeat_session("shared".into());
    state.invalidate_scope(&second);
    assert!(state.pending_close.is_empty());
    state.invalidate(&PromptSource::Heartbeat);
    assert_eq!(state.pending_close, HashSet::from(["shared".into()]));

    state.install_scope_session(first.clone(), "shared".into());
    state.install_scope_session(first, "shared".into());
    assert!(state.pending_close.is_empty());
}

#[tokio::test]
async fn absent_close_capability_discards_queue_without_a_wire_request() {
    let observer = observer::ObserverHandle::in_process();
    let mut agent = start_agent(&adapter_config(json!("absent")), &observer).await;
    agent.state.install_heartbeat_session("abandoned".into());
    agent.state.invalidate(&PromptSource::Heartbeat);
    assert_eq!(agent.state.pending_close.len(), 1);

    let mut result = run_heartbeat(agent, tests::make_prompt_context_no_owner()).await;

    assert!(matches!(
        result.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert!(result.agent.state.pending_close.is_empty());
    assert_eq!(
        wire_requests(&observer)
            .iter()
            .map(|request| request["method"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["initialize", "session/new", "session/prompt"],
    );
    result.agent.acp.shutdown().await;
}

#[tokio::test]
async fn max_turn_rotation_closes_the_old_session_before_the_next_turn() {
    let observer = observer::ObserverHandle::in_process();
    let agent = start_agent(&adapter_config(json!({})), &observer).await;
    let mut ctx = tests::make_prompt_context_no_owner();
    ctx.max_turns_per_session = 1;
    let result = run_heartbeat(agent, ctx).await;
    assert!(matches!(
        result.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert_eq!(
        result.agent.state.pending_close,
        HashSet::from(["session-1".into()])
    );
    assert_eq!(
        result.agent.abandoned_session_close_budget(),
        Duration::from_secs(10)
    );

    let mut result = run_heartbeat(result.agent, tests::make_prompt_context_no_owner()).await;

    assert!(matches!(
        result.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert_eq!(
        result.agent.state.heartbeat_session.as_deref(),
        Some("session-2")
    );
    assert!(result.agent.state.pending_close.is_empty());
    let requests = wire_requests(&observer);
    assert_eq!(requests[3]["method"], "session/close");
    assert_eq!(requests[3]["params"], json!({"sessionId": "session-1"}));
    assert_eq!(requests[4]["method"], "session/new");
    result.agent.acp.shutdown().await;
}

#[tokio::test]
async fn close_rejection_and_timeout_do_not_block_the_next_turn() {
    let observer = observer::ObserverHandle::in_process();
    let mut agent = start_agent(&adapter_config(json!({})), &observer).await;
    agent.state.install_heartbeat_session("times-out".into());
    agent.state.install_heartbeat_session("rejected".into());
    agent.state.invalidate(&PromptSource::Heartbeat);

    let start = tokio::time::Instant::now();
    let mut result = run_heartbeat(agent, tests::make_prompt_context_no_owner()).await;

    assert!(start.elapsed() >= Duration::from_secs(10));
    assert!(matches!(
        result.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert!(result.agent.state.pending_close.is_empty());
    let requests = wire_requests(&observer);
    assert_eq!(requests.len(), 5);
    let close_ids: HashSet<_> = requests[1..3]
        .iter()
        .map(|request| {
            assert_eq!(request["method"], "session/close");
            request["params"]["sessionId"].as_str().unwrap()
        })
        .collect();
    assert_eq!(close_ids, HashSet::from(["times-out", "rejected"]));
    assert_eq!(requests[3]["method"], "session/new");
    assert_eq!(requests[4]["method"], "session/prompt");
    result.agent.acp.shutdown().await;
}

#[tokio::test]
async fn failed_session_setup_queues_the_id_returned_by_session_new() {
    let observer = observer::ObserverHandle::in_process();
    let mut agent = start_agent(&adapter_config(json!({})), &observer).await;
    agent.agent_name = "goose".into();
    let mut ctx = tests::make_prompt_context_no_owner();
    ctx.system_prompt = Some("test system prompt".into());

    let mut result = run_heartbeat(agent, ctx).await;

    assert!(matches!(
        result.outcome,
        PromptOutcome::Error(AcpError::AgentError { .. })
    ));
    assert!(result.agent.state.heartbeat_session.is_none());
    assert_eq!(
        result.agent.state.pending_close,
        HashSet::from(["session-1".into()])
    );
    result.agent.acp.shutdown().await;
}

#[tokio::test]
async fn respawn_discards_the_old_subprocess_close_queue() {
    let old_observer = observer::ObserverHandle::in_process();
    let config = adapter_config(json!({}));
    let mut old_agent = start_agent(&config, &old_observer).await;
    old_agent
        .state
        .install_heartbeat_session("old-process".into());
    old_agent.state.invalidate(&PromptSource::Heartbeat);
    assert_eq!(old_agent.state.pending_close.len(), 1);
    let observer = observer::ObserverHandle::in_process();
    let (respawn_tx, mut respawn_rx) = mpsc::channel(1);
    let mut respawn_tasks = JoinSet::new();
    let mut slot = crate::SlotCircuit {
        crash_times: Vec::new(),
        open_until: None,
        respawn_in_flight: false,
    };
    assert!(crate::spawn_respawn_task(
        old_agent,
        &config,
        &mut slot,
        &respawn_tx,
        &mut respawn_tasks,
        Some(observer.clone()),
    ));
    let agent = timeout(Duration::from_secs(10), respawn_rx.recv())
        .await
        .expect("respawn should finish")
        .expect("respawn result")
        .into_agent(&config)
        .expect("start replacement process");
    assert!(agent.state.pending_close.is_empty());
    assert!(agent.acp.session_close_supported());

    let mut result = run_heartbeat(agent, tests::make_prompt_context_no_owner()).await;

    assert!(matches!(
        result.outcome,
        PromptOutcome::Ok(StopReason::EndTurn)
    ));
    assert!(wire_requests(&observer)
        .iter()
        .all(|request| request["method"] != "session/close"));
    result.agent.acp.shutdown().await;
    respawn_tasks
        .join_next()
        .await
        .expect("finished respawn task")
        .unwrap();
}
