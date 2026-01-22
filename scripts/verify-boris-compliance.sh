#!/bin/bash
# Boris Cherny Compliance Verification Script
# Validates that the repository follows Boris Cherny's agentic patterns

# Change to script directory's parent (project root) if running from elsewhere
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Boris Cherny Compliance Check ==="
echo "Project: $PROJECT_ROOT"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

ERRORS=0
WARNINGS=0
PASSES=0

# Function to check and report
check() {
    local condition="$1"
    local message="$2"
    local is_warning="${3:-false}"

    if eval "$condition" 2>/dev/null; then
        echo -e "${GREEN}[PASS]${NC} $message"
        PASSES=$((PASSES + 1))
    else
        if [ "$is_warning" = "true" ]; then
            echo -e "${YELLOW}[WARN]${NC} $message"
            WARNINGS=$((WARNINGS + 1))
        else
            echo -e "${RED}[FAIL]${NC} $message"
            ERRORS=$((ERRORS + 1))
        fi
    fi
}

echo "=== Layer 1: Claude Code Configuration ==="
echo ""

# 1. Check CLAUDE.md exists
check "[ -f 'CLAUDE.md' ]" "CLAUDE.md exists at root"

# 2. Check CLAUDE.md has Common Mistakes section
check "grep -q 'Common Mistakes' CLAUDE.md 2>/dev/null" "CLAUDE.md has Common Mistakes section"

# 3. Check CLAUDE.md has NEVER-DO section
check "grep -q 'NEVER-DO' CLAUDE.md 2>/dev/null" "CLAUDE.md has NEVER-DO section"

# 4. Check .claude/agents/ directory exists with files
check "[ -d '.claude/agents' ] && [ -n \"\$(ls -A .claude/agents/*.md 2>/dev/null)\" ]" ".claude/agents/ contains agent files"

# 5. Check agents have YAML frontmatter
AGENTS_WITH_FRONTMATTER=0
TOTAL_AGENTS=0
for agent in .claude/agents/*.md; do
    if [ -f "$agent" ]; then
        TOTAL_AGENTS=$((TOTAL_AGENTS + 1))
        if head -1 "$agent" | grep -q "^---"; then
            AGENTS_WITH_FRONTMATTER=$((AGENTS_WITH_FRONTMATTER + 1))
        fi
    fi
done
check "[ $AGENTS_WITH_FRONTMATTER -eq $TOTAL_AGENTS ]" "All agents have YAML frontmatter ($AGENTS_WITH_FRONTMATTER/$TOTAL_AGENTS)"

# 6. Check agents have tools: array
AGENTS_WITH_TOOLS=0
for agent in .claude/agents/*.md; do
    if [ -f "$agent" ] && grep -q "^tools:" "$agent"; then
        AGENTS_WITH_TOOLS=$((AGENTS_WITH_TOOLS + 1))
    fi
done
check "[ $AGENTS_WITH_TOOLS -eq $TOTAL_AGENTS ]" "All agents have tools: restriction ($AGENTS_WITH_TOOLS/$TOTAL_AGENTS)"

# 7. Check agents have model: specified
AGENTS_WITH_MODEL=0
for agent in .claude/agents/*.md; do
    if [ -f "$agent" ] && grep -q "^model:" "$agent"; then
        AGENTS_WITH_MODEL=$((AGENTS_WITH_MODEL + 1))
    fi
done
check "[ $AGENTS_WITH_MODEL -eq $TOTAL_AGENTS ]" "All agents have model: specified ($AGENTS_WITH_MODEL/$TOTAL_AGENTS)"

# 8. Check .claude/commands/ exists with files
check "[ -d '.claude/commands' ] && [ -n \"\$(ls -A .claude/commands/*.md 2>/dev/null)\" ]" ".claude/commands/ contains command files"

# 9. Check commands have YAML frontmatter with description
COMMANDS_WITH_DESC=0
TOTAL_COMMANDS=0
for cmd in .claude/commands/*.md; do
    if [ -f "$cmd" ]; then
        TOTAL_COMMANDS=$((TOTAL_COMMANDS + 1))
        if head -5 "$cmd" | grep -q "description:"; then
            COMMANDS_WITH_DESC=$((COMMANDS_WITH_DESC + 1))
        fi
    fi
done
check "[ $COMMANDS_WITH_DESC -eq $TOTAL_COMMANDS ]" "All commands have description: frontmatter ($COMMANDS_WITH_DESC/$TOTAL_COMMANDS)"

# 10. Check settings.json exists
check "[ -f '.claude/settings.json' ]" ".claude/settings.json exists"

# 11. Check settings.json has hooks section
check "grep -q '\"hooks\"' .claude/settings.json 2>/dev/null" "settings.json has hooks section"

# 12. Check hooks have type: command
check "grep -q '\"type\".*:.*\"command\"' .claude/settings.json 2>/dev/null" "Hooks use type: command"

# 13. Check PreToolUse hooks exist
check "[ -d '.claude/hooks/PreToolUse' ] && [ -n \"\$(ls -A .claude/hooks/PreToolUse/*.sh 2>/dev/null)\" ]" "PreToolUse hooks exist"

# 14. Check PostToolUse hooks exist
check "[ -d '.claude/hooks/PostToolUse' ] && [ -n \"\$(ls -A .claude/hooks/PostToolUse/*.sh 2>/dev/null)\" ]" "PostToolUse hooks exist"

# 15. Check hooks are executable
HOOKS_EXECUTABLE=true
for hook in .claude/hooks/*/*.sh; do
    if [ -f "$hook" ] && [ ! -x "$hook" ]; then
        HOOKS_EXECUTABLE=false
        break
    fi
done
check "$HOOKS_EXECUTABLE" "All hook scripts are executable"

# 16. Check .mcp.json exists
check "[ -f '.mcp.json' ]" ".mcp.json exists"

# 17. Check verification scripts exist
check "[ -d 'scripts' ] && [ -n \"\$(ls scripts/verify-*.sh 2>/dev/null)\" ]" "Verification scripts exist"

echo ""
echo "=== Layer 2: Application Code (Optional) ==="
echo ""

# These are optional but good to have
check "[ -d 'src' ]" "src/ directory exists" "true"
check "[ -d 'tests' ]" "tests/ directory exists" "true"
check "[ -f 'src/config.py' ]" "Configuration module exists" "true"

echo ""
echo "=== Layer 3: Runtime BORIS Features ==="
echo ""

# Middleware directory exists
check "[ -d 'src/middleware' ]" "src/middleware/ directory exists"

# All 3 middleware files exist
check "[ -f 'src/middleware/__init__.py' ]" "Middleware __init__.py exists"
check "[ -f 'src/middleware/autonomy.py' ]" "Autonomy middleware exists"
check "[ -f 'src/middleware/cost_tracking.py' ]" "Cost tracking middleware exists"
check "[ -f 'src/middleware/audit.py' ]" "Audit logging middleware exists"

# BaseAgent imports middleware
check "grep -q 'get_autonomy_middleware\|get_cost_middleware\|get_audit_middleware' src/agents/base.py 2>/dev/null" "BaseAgent imports middleware functions"

# Middleware is integrated in BaseAgent.run()
check "grep -q 'audit.log_agent_start' src/agents/base.py 2>/dev/null" "BaseAgent.run() has audit logging"

# Middleware is integrated in BaseAgent.call_llm()
check "grep -q 'cost_middleware.after_llm_call' src/agents/base.py 2>/dev/null" "BaseAgent.call_llm() has cost tracking"

# Middleware is integrated in BaseAgent.present_options()
check "grep -q 'autonomy.mark_options_presented' src/agents/base.py 2>/dev/null" "BaseAgent.present_options() has autonomy tracking"

# Middleware tests exist
check "[ -d 'tests/test_middleware' ]" "tests/test_middleware/ directory exists"
check "[ -f 'tests/test_middleware/test_autonomy.py' ]" "Autonomy middleware tests exist"
check "[ -f 'tests/test_middleware/test_cost_tracking.py' ]" "Cost tracking middleware tests exist"
check "[ -f 'tests/test_middleware/test_audit.py' ]" "Audit logging middleware tests exist"

echo ""
echo "=== Summary ==="
echo ""
echo -e "Passes:   ${GREEN}$PASSES${NC}"
echo -e "Warnings: ${YELLOW}$WARNINGS${NC}"
echo -e "Failures: ${RED}$ERRORS${NC}"
echo ""

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}Boris Cherny Compliance: PASSED${NC}"
    exit 0
else
    echo -e "${RED}Boris Cherny Compliance: FAILED${NC}"
    echo ""
    echo "Fix the failures above to achieve compliance."
    exit 1
fi
