# The Analyst - Setup Guide

Complete setup instructions for The Analyst analytics platform.

## Prerequisites

### Required
- **Python 3.11+** - Core language runtime
- **PostgreSQL 15+** - Database (or Docker)
- **uv** - Fast Python package manager

### Optional
- **Docker & Docker Compose** - For containerized setup
- **HuggingFace account** - For Arabic NLP models (MARBERT)

---

## Installation Methods

### Method 1: Local Development (Recommended)

#### Step 1: Install Python 3.11+

```bash
# macOS (Homebrew)
brew install python@3.11

# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv

# Verify installation
python3.11 --version
```

#### Step 2: Install uv (Package Manager)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv

# Verify
uv --version
```

#### Step 3: Clone Repository

```bash
git clone <repository-url>
cd the-analyst
```

#### Step 4: Create Virtual Environment

```bash
uv venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

#### Step 5: Install Dependencies

```bash
# Core + development dependencies
uv pip install -e ".[dev]"

# Optional: Arabic NLP support
uv pip install -e ".[arabic]"

# Optional: PDF report generation
uv pip install -e ".[reports]"

# All optional dependencies
uv pip install -e ".[all]"
```

#### Step 6: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your values (required):
#   ANTHROPIC_API_KEY=sk-ant-...
#   DATABASE_URL=postgresql://user:pass@localhost:5432/analyst
```

#### Step 7: Setup Database

```bash
# Option A: Use Docker (recommended)
docker-compose up -d db

# Option B: Use existing PostgreSQL
# Create database first, then run:
./scripts/setup-db.sh
```

#### Step 8: Verify Installation

```bash
# Test imports
python -c "from src.config import get_settings; print('Config: OK')"
python -c "from src.agents.base import BaseAgent; print('Agents: OK')"

# Run test suite
pytest

# Check Boris compliance
./scripts/verify-boris-compliance.sh
```

---

### Method 2: Docker Setup

#### Step 1: Install Docker

```bash
# macOS
brew install --cask docker

# Ubuntu
sudo apt install docker.io docker-compose
```

#### Step 2: Configure Environment

```bash
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY
```

#### Step 3: Build and Run

```bash
# Build image
docker-compose build

# Start services (app + PostgreSQL)
docker-compose up -d

# View logs
docker-compose logs -f analyst

# Stop services
docker-compose down
```

---

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `ANTHROPIC_API_KEY` | Claude API key | `sk-ant-api03-...` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@localhost:5432/analyst` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGINGFACE_TOKEN` | - | For Arabic NLP models (MARBERT) |
| `LOG_LEVEL` | `INFO` | Logging verbosity (DEBUG, INFO, WARNING, ERROR) |
| `COST_ALERT_THRESHOLD` | `1.0` | USD threshold for API cost alerts |
| `MAX_ROWS_WITHOUT_CONFIRM` | `10000` | Row count requiring confirmation |
| `OBSIDIAN_VAULT_PATH` | - | Path to Obsidian vault for notifications |
| `ENABLE_DESKTOP_NOTIFICATIONS` | `true` | Enable desktop notifications |
| `DATA_PATH` | `./data` | Base path for data directories |
| `OUTPUT_PATH` | `./data/outputs` | Path for generated outputs |

---

## IDE Setup

### VS Code (Recommended)

**Recommended Extensions:**
- Python (`ms-python.python`)
- Pylance (`ms-python.vscode-pylance`)
- Ruff (`charliermarsh.ruff`)
- Black Formatter (`ms-python.black-formatter`)

**Settings** (`.vscode/settings.json`):
```json
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.analysis.typeCheckingMode": "strict",
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests"]
}
```

### PyCharm

1. Open project directory
2. Configure interpreter: File > Settings > Project > Python Interpreter > `.venv/bin/python`
3. Enable Black: Settings > Tools > Black > Enable
4. Configure pytest: Settings > Tools > Python Integrated Tools > Default test runner: pytest

### Cursor

Uses same settings as VS Code. Recommended extensions apply.

---

## Verification Checklist

After setup, verify everything works:

```bash
# 1. Environment check
python -c "
from src.config import get_settings
s = get_settings()
print(f'API Key: {s.anthropic_api_key[:15]}...')
print(f'Database: {s.database_url[:30]}...')
print('Environment: OK')
"

# 2. Import check
python -c "
from src.agents.base import BaseAgent
from src.agents.retrieval import RetrievalAgent
from src.agents.statistical import StatisticalAgent
from src.orchestrator.main import Orchestrator
print('Imports: OK')
"

# 3. Run tests
pytest -v

# 4. Boris compliance
./scripts/verify-boris-compliance.sh

# 5. Type checking
mypy src/

# 6. Code formatting
black --check src/ tests/
ruff check src/ tests/
```

**All checks should pass before starting development.**

---

## Database Setup Details

### PostgreSQL Schema

The application creates these tables automatically:

| Table | Purpose |
|-------|---------|
| `analysis_history` | Track all analyses (session, type, status, results) |
| `data_checksums` | Data integrity verification |
| `user_preferences` | Persistent user settings |
| `workflow_states` | Workflow recovery and state persistence |

### Manual Database Creation

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database and user
CREATE DATABASE analyst;
CREATE USER analyst_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE analyst TO analyst_user;

# Exit
\q

# Update .env
DATABASE_URL=postgresql://analyst_user:your_password@localhost:5432/analyst
```

---

## Troubleshooting Setup

### Python Version Issues

```bash
# Check Python version
python --version

# If wrong version, use pyenv
pyenv install 3.11.7
pyenv local 3.11.7
```

### Database Connection Refused

```bash
# Check PostgreSQL is running
docker-compose ps

# Or for local PostgreSQL
brew services list  # macOS
systemctl status postgresql  # Linux

# Test connection
psql -h localhost -U analyst -d analyst
```

### Missing System Dependencies

```bash
# For WeasyPrint (PDF generation) on macOS
brew install pango cairo gdk-pixbuf libffi

# On Ubuntu
sudo apt install libpango-1.0-0 libcairo2 libgdk-pixbuf2.0-0
```

### Virtual Environment Not Activating

```bash
# Recreate environment
rm -rf .venv
uv venv .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

---

## Next Steps

1. **Read** `CLAUDE.md` for project philosophy and constraints
2. **Explore** slash commands: `/analyze`, `/ingest`, `/report`
3. **Review** agent definitions in `.claude/agents/`
4. **Run** verification: `./scripts/verify-boris-compliance.sh`
5. **Start** the orchestrator: `python -m src.orchestrator.main`

---

## Getting Help

- **CLAUDE.md** - Project context and development guidelines
- **README.md** - Quick reference and feature overview
- **memory/context.md** - Project history and decisions
- **.claude/agents/** - Individual agent documentation
