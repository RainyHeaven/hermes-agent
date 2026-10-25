# 포크 커스텀 목록 (RainyHeaven/hermes-agent)

upstream(NousResearch/hermes-agent) 안정판 태그 위에 얹는 이 포크 고유 변경의 단일 목록이다.
버전업 때 병합·충돌 해소·검증은 이 문서를 기준으로 한다. 항목을 추가·제거하면 이 문서도 같은 커밋에서 갱신한다.

- 기준 upstream: v0.22.2
- 원칙: 코어 수정은 최소한의 연결 지점으로 줄이고, 가능하면 plugin·config·upstream PR로 옮긴다.
- 런타임 설정(`~/.hermes/config.yaml`, `.env`, 스킬, 세션 DB)은 저장소 밖에 있으며 이 목록의 대상이 아니다.

## 현재 유지 중인 커스텀

### C1. 로컬 런타임 파일 커밋 방지 (`.gitignore`)
- 의도: 저장소 안에 실수로 생긴 config·auth·세션·메모리 파일이 public 포크에 올라가지 않게 한다.
- 파일: `.gitignore` (12줄 추가)
- 확인: `git check-ignore -v config.yaml auth.json sessions/x .hermes/x`
- 제거 조건: 없음(충돌 위험 낮음).

### C2. Discord 채널별 메시지 배치 지연
- 의도: 채널마다 `channel_text_batch_delays` / `channel_text_batch_split_delays`로 메시지 묶음 지연을 다르게 둔다. 정확한 채널 → 부모 채널 순으로 찾는다.
- 파일: `plugins/platforms/discord/adapter.py` (upstream 훅 `_text_batch_delay_for`에 연결), `tests/gateway/test_text_batching.py`
- 설정: `discord.channel_text_batch_delays`, `discord.channel_text_batch_split_delays` (upstream config_loader가 extra로 전달하므로 gateway/config.py 수정 불필요)
- 종료 시 처리: `cancel_background_tasks`가 아직 지연 대기 중인 배치를 즉시 dispatch한다. 채널 지연(예: 15초)이 종료 flush 기한(5초 미만)보다 길면 기한 초과 cancel로 메시지가 사라지기 때문이다.
- 확인 테스트: `tests/gateway/test_text_batching.py`, `tests/gateway/test_discord_channel_delay_shutdown.py`
- 제거 조건: upstream에 같은 기능이 들어가면 제거. upstream은 배치 지연 상한(2~4초) 방침이라 PR보다 기능 제안 이슈를 먼저 연다. 제출 전 테스트의 wall-clock sleep을 upstream health 규칙(HX013)에 맞게 바꿔야 한다.

### C3. 컨텍스트 rollover (누적 압축 후 새 세션·새 에이전트로 이어가기)
- 의도: CLI/TUI 세션에서 full compaction이 `compression.rollover_after`회 쌓이면 안전한 턴 경계에서 인계 자료를 저장하고 새 세션으로 같은 작업을 이어간다. 기본 0=꺼짐.
- 집계 기준: 실제 요약이 만들어진 full compaction만 센다. 요약 생성 실패로 대체 문구만 남은 압축(`_last_summary_fallback_used`)은 세지 않는다(`agent/conversation_compression.py::_commit_compaction`의 `_counts_as_full`).
- 새 모듈(충돌 위험 낮음): `agent/context_rollover*.py`, `hermes_cli/cli_context_rollover.py`, `hermes_state_rollover.py`, `tui_gateway/session_rollover.py`
- 코어 연결 지점(충돌 위험 높음): `agent/conversation_compression.py`, `agent/conversation_loop.py`, `agent/turn_context.py`, `hermes_cli/cli_agent_setup_mixin.py`, `hermes_cli/cli_chat_turn_mixin.py`, `hermes_cli/cli_tui_runtime_mixin.py`, `hermes_cli/config_defaults.py`, `hermes_state_compression.py`, `hermes_state_messages.py`, `hermes_state_sessions.py`, `tui_gateway/{methods_prompt,prompt_turn,server,session_auto_continue}.py`
- 확인 테스트: `tests/agent/test_context_rollover*.py`, `tests/agent/test_compression_rollover_counter.py`, `tests/hermes_cli/test_cli_context_rollover.py`, `tests/hermes_state/test_session_rollover.py`, `tests/tui_gateway/test_session_rollover.py`
- 제거·축소 조건: 코어 연결 지점을 plugin hook 또는 최소 seam으로 줄일 수 있으면 이전. upstream에 유사 기능이 생기면 대체 검토.

### C4. plugin_validate 테스트의 macOS 진단 suffix 허용 (테스트 전용)
- 의도: macOS에서는 가용성 detail 뒤에 `,<진단>`이 붙어 upstream 테스트가 실패한다. 상태 토큰만 비교하도록 3줄 보정한다. 제품 코드 변경 없음.
- 파일: `tests/hermes_cli/test_plugin_validate.py`
- 제거 조건: upstream에서 같은 보정이 들어가면 제거. upstream PR #142824 제출(2026-10-22).

### C5. 대시보드 PTY 재부착 키의 세션 계보 일치 (`/api/pty` keep-alive)
- 의도: 터미널을 resume=A로 연 뒤 그 안에서 압축·rollover로 후속 세션 B가 생기면, 같은 attach 토큰으로 다시 붙을 때 키가 `…\0B`로 바뀌어 살아 있는(진행 중 턴일 수 있는) 같은 터미널이 4409로 닫히고 새 PTY가 떴다. 같은 attach·profile 접두의 살아 있는 키 중 resume 세그먼트의 최신 후속이 지금 요청의 최신 후속과 같은 것이 정확히 하나면 그 터미널로 다시 붙는다. 없거나 여럿이거나 조회 실패(warning 로그)면 기존 동작. lease를 대시보드 in-memory gateway가 잡는 배치에서는 holder_pid로 터미널을 식별할 수 없어서 계보를 쓴다.
- 파일: `hermes_cli/web_routers/chat_ws.py`(`_latest_descendant_for`, `_lineage_attach_key`, keep-alive 경로 한 줄), `hermes_cli/pty_session.py`(`PtySessionRegistry.live_keys_with_prefix`)
- 확인 테스트: `tests/hermes_cli/test_pty_resume_lineage_ws.py`(하네스 7개, 실제 세션 DB 2개)
- 알려진 한계: 터미널 안에서 `/resume`으로 다른 대화로 옮긴 뒤에도 같은 토큰의 계보 요청은 그 터미널에 붙는다(같은 키 재부착에도 원래 있던 한계). fresh로 연 키(resume 세그먼트 없음)는 대상이 아니다.
- 제거 조건: upstream에 같은 수정이 들어가면 제거. upstream PR 후보(재현 시험 포함).

## 알려진 환경 의존 실패 (커스텀과 무관)

- `tests/hermes_state/` 의 state_db repair/maintenance 계열 63건은 v0.22.2 + 커스텀 기준선에서도 동일하게 실패한다(2026-10-22, 이 기기·test-venv). 버전업 검증에서는 실패 ID 집합을 기준선 clone과 비교해 새 실패만 본다.

## 포크에서 제거한 커스텀 (2026-10-22)

| 항목 | 제거 이유 | 보관 위치 |
|---|---|---|
| GitHub 스킬의 `~/Git/.env` 토큰 대체 경로 | 1순위 인증인 `gh auth`가 로그인돼 있어 대체 경로가 쓰이지 않음. 번들 스킬 수정은 매 버전업 충돌 원인 | 로컬 보관 패치 |
| v16 세션 마이그레이션 추정 제거 | 이 기기 DB가 이미 v31이라 다시 실행되지 않음 | upstream PR #142823 제출 |
| Codex 이미지 회귀 테스트 3종 | upstream v0.22.2가 같은 native 경로를 구현함. 테스트만 남은 차이(포크 전용 host_model 관련이라 upstream 제출 안 함) | 로컬 보관 패치 |

보관 패치: `~/.hermes/reports/fork-slimming-20261022/upstream-candidates/` (저장소 밖, 비공개)

## 버전업 절차 요약

상세 절차는 Hermes 스킬 `hermes-agent`의 `references/fork-upgrade-runbook.md`를 따른다.

1. 대상은 upstream 안정판 태그로 고정한다(main 아님). 다른 작업과 섞지 않는다. 새 태그는 바로 받지 않고 어느 정도 안정화된 뒤 받는다(사용자 결정 2026-10-22). 기준: 출시 후 2주 이상 지났고, 그 사이 같은 영역의 긴급 수정이 이어지지 않으며, 구조 변경이 있으면 커스텀 영향 범위를 먼저 확인한다.

## 보류 중인 버전업 메모

- v0.22.4 (2026-10-21): 보류. 클래식 CLI가 공유 gateway 클라이언트(`hermes_cli/gateway_agent.py`)로 바뀌어 턴이 gateway 프로세스에서 실행된다. 이에 따라 C3의 CLI 어댑터(`hermes_cli/cli_context_rollover.py`, `_agent_build_kwargs`)는 그대로 쓸 수 없고, TUI gateway 경로 하나로 CLI까지 처리하도록 재설계가 필요하다. `_make_agent`는 `tui_gateway/agent_factory.py`로 이동했다(exact_route 이식 패치 보관). 같은 버전의 `post_compaction` plugin hook으로 압축 횟수 집계를 옮기고, 집계·인계 기록을 rollover 전용 테이블로 옮기면 `hermes_state_{compression,messages,sessions}.py` 수정을 없앨 수 있다. 다음 안정판에서 함께 진행한다.
2. 개발 clone `~/Git/hermes-agent-dev`에서 태그 위로 병합하고, 이 문서의 항목별 확인 테스트를 돌린다. `git rerere`가 켜져 있어 이전 충돌 해소가 재사용된다.
3. 설치본 `~/.hermes/hermes-agent`는 개발 clone에서 fetch 후 ff-only로만 반영한다.
4. 반영 후 번들 스킬 동기화 결과(`hermes skills list-modified`)와 config 보존 키를 확인한다. 업데이트된 번들 스킬을 예전 버전으로 되돌리지 않는다.
