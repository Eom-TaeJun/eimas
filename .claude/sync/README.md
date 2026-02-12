# EIMAS Team Communication Directory

## Team Structure

### Team Alpha (Development - tjeom01@gmail.com)
- **Role**: Feature development, refactoring, implementation
- **Agents**: alpha-lead, alpha-backend, alpha-frontend, alpha-refactor

### Team Beta (QA & Security - eomtj2001@gmail.com)
- **Role**: Code review, testing, security audit, performance validation
- **Agents**: beta-lead, beta-qa, beta-security, beta-performance

## Directory Structure

```
.claude/sync/
├── tasks/
│   ├── pending/          # Tasks waiting for Team Beta
│   ├── in_progress/      # Currently being worked on
│   └── completed/        # Finished tasks with results
├── messages/
│   ├── alpha_to_beta/    # Messages from Alpha to Beta
│   └── beta_to_alpha/    # Messages from Beta to Alpha
├── status/
│   ├── alpha_status.json # Team Alpha current status
│   └── beta_status.json  # Team Beta current status
└── locks/                # File locks for concurrent access
```

## Task File Format

```json
{
  "id": "task_TIMESTAMP_type",
  "from_team": "alpha",
  "to_team": "beta",
  "type": "code_review|testing|security_audit|performance_test",
  "priority": "low|medium|high|critical",
  "files": ["path/to/file1.py", "path/to/file2.py"],
  "description": "What needs to be done",
  "context": {
    "track": "B|C|E|F",
    "milestone": "description",
    "acceptance_criteria": ["criterion1", "criterion2"]
  },
  "created_at": "ISO 8601 timestamp",
  "assigned_to": null
}
```

## Workflow

### Team Alpha (Development)
1. Implement feature/refactoring
2. Create task in `tasks/pending/`
3. Update `status/alpha_status.json`
4. Wait for Team Beta completion

### Team Beta (QA/Security)
1. Monitor `tasks/pending/`
2. Pick up task → move to `in_progress/`
3. Execute review/testing
4. Create result file in `completed/`
5. Send feedback via `messages/beta_to_alpha/`

## Task Types

### code_review
- Static analysis
- Code quality check
- Best practices validation
- Documentation review

### testing
- Unit tests execution
- Integration tests
- Regression tests
- Test coverage analysis

### security_audit
- Vulnerability scan
- Dependency check
- Security best practices
- API key/secret validation

### performance_test
- Execution time benchmarks
- Memory profiling
- Database query optimization
- Load testing

## Priority Levels

- **critical**: Blocks deployment, must fix immediately
- **high**: Important for quality, fix before merge
- **medium**: Should fix, can be deferred
- **low**: Nice to have, optional

## Status Codes

- `ready`: Waiting for work
- `working`: Currently processing task
- `blocked`: Cannot proceed, needs input
- `complete`: Task finished, results available
