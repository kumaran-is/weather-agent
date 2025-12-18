"""
Level 1 Setup Verification Script
Ensures all components are properly configured for Level 1 (ReAct Agent + HITL).

Tests:
1. Python 3.13+ installed
2. Environment variables set (Docker + HTTP transport)
3. OpenAI API connection
4. Anthropic API connection (optional)
5. All Docker containers running (Weather MCP + Hurricane MCP + Weather AI API)
6. MCP health endpoints accessible
7. LangSmith tracing enabled
8. Docker installed and running
9. LangGraph CLI installed (for LangSmith Studio)
"""

import os
import sys
from pathlib import Path

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    if Path(".env").exists():
        load_dotenv()
        print("📄 Loaded .env file")
    else:
        print("⚠️  No .env file found - using .env.template as reference")
        print("   Copy .env.template to .env and add your API keys\n")
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env loading\n")

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def print_header(text):
    """Print section header"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*70}{RESET}\n")

def check_python_version():
    """Verify Python 3.13+"""
    print("🔎 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 13:
        print(f"{GREEN}✅ Python 3.13+ detected:{RESET} {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"{RED}❌ Python 3.13+ required, found:{RESET} {version.major}.{version.minor}.{version.micro}")
        print(f"{YELLOW}   Install Python 3.13: https://www.python.org/downloads/{RESET}")
        return False

def check_env_vars():
    """Verify required environment variables"""
    print("\n🔎 Checking environment variables...")

    required_vars = {
        "OPENAI_API_KEY": False,  # Required
        "LANGCHAIN_API_KEY": False,  # Required
        "LANGCHAIN_TRACING_V2": False,  # Required
        "MCP_WEATHER_SERVER_URL": False,  # Required (Docker + HTTP)
        "MCP_HURRICANE_SERVER_URL": False,  # Required (Docker + HTTP)
    }

    optional_vars = {
        "ANTHROPIC_API_KEY": "Optional - for Claude models in future levels",
        "MCP_WEATHER_SERVER_ENABLED": "Optional - defaults to true",
        "MCP_HURRICANE_SERVER_ENABLED": "Optional - defaults to true",
        "LOG_LEVEL": "Optional - defaults to 'info' (options: debug, info, warning, error)",
        "LANGCHAIN_ENDPOINT": "Optional - defaults to 'https://api.smith.langchain.com'",
        "LANGCHAIN_PROJECT": "Optional - defaults to 'weather-ai-agent-service'"
    }

    all_present = True

    # Check required vars
    for var in required_vars:
        value = os.getenv(var)
        if value:
            # Mask sensitive values
            if "KEY" in var or "TOKEN" in var:
                masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
                print(f"{GREEN}✅ {var}{RESET} = {masked}")
            else:
                print(f"{GREEN}✅ {var}{RESET} = {value}")
        else:
            print(f"{RED}❌ {var}{RESET} is missing")
            all_present = False

    # Check optional vars
    print(f"\n{BLUE}Optional Variables:{RESET}")
    for var, description in optional_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:8] + "..." + value[-4:] if len(value) > 12 else "***"
            print(f"{GREEN}✅ {var}{RESET} = {masked} ({description})")
        else:
            print(f"{YELLOW}⚠️  {var}{RESET} not set ({description})")

    return all_present

def check_openai_connection():
    """Test OpenAI API connection"""
    print("\n🔎 Checking OpenAI API connection...")

    if not os.getenv("OPENAI_API_KEY"):
        print(f"{YELLOW}⚠️  OPENAI_API_KEY not set, skipping test{RESET}")
        return False

    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=50)
        response = llm.invoke("Say 'Setup verified!'")

        if "Setup verified" in response.content or "setup verified" in response.content.lower():
            print(f"{GREEN}✅ OpenAI API connection successful{RESET}")
            print(f"   Response: {response.content[:50]}")
            return True
        else:
            print(f"{YELLOW}⚠️  OpenAI API response unexpected{RESET}")
            print(f"   Response: {response.content[:100]}")
            return True  # Still count as success if we got a response
    except Exception as e:
        print(f"{RED}❌ OpenAI API connection failed:{RESET} {str(e)[:150]}")
        print(f"{YELLOW}   Check your OPENAI_API_KEY at: https://platform.openai.com/api-keys{RESET}")
        return False

def check_anthropic_connection():
    """Test Anthropic API connection (optional)"""
    print("\n🔎 Checking Anthropic API connection (optional)...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"{YELLOW}⚠️  ANTHROPIC_API_KEY not set (optional){RESET}")
        return True  # Don't fail if optional

    try:
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0, max_tokens=50)
        response = llm.invoke("Say 'Setup verified!'")

        if "Setup verified" in response.content or "setup verified" in response.content.lower():
            print(f"{GREEN}✅ Anthropic API connection successful{RESET}")
            return True
        else:
            print(f"{YELLOW}⚠️  Anthropic API response unexpected (optional){RESET}")
            return True  # Don't fail
    except Exception as e:
        print(f"{YELLOW}⚠️  Anthropic API connection failed (optional):{RESET} {str(e)[:150]}")
        return True  # Don't fail if optional

def check_mcp_servers():
    """Check MCP Docker containers and health endpoints"""
    print("\n🔎 Checking Docker containers...")

    import subprocess

    weather_url = os.getenv("MCP_WEATHER_SERVER_URL", "http://localhost:8080")
    hurricane_url = os.getenv("MCP_HURRICANE_SERVER_URL", "http://localhost:8081")
    api_url = "http://localhost:8000"

    all_good = True

    # Check if docker-compose services are running
    try:
        result = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            containers = result.stdout

            # Check Weather MCP container
            if "weather-mcp-server" in containers:
                status = [line for line in containers.split('\n') if 'weather-mcp-server' in line]
                if status and 'Up' in status[0]:
                    print(f"{GREEN}✅ Weather MCP Server container running{RESET}")
                    print("   Container: weather-mcp-server")
                    print(f"   URL: {weather_url}")
                else:
                    print(f"{YELLOW}⚠️  Weather MCP Server container exists but not running{RESET}")
                    all_good = False
            else:
                print(f"{YELLOW}⚠️  Weather MCP Server container not found{RESET}")
                print("   Run: docker-compose up -d")
                all_good = False

            # Check Hurricane MCP container
            if "hurricane-tracker-mcp" in containers:
                status = [line for line in containers.split('\n') if 'hurricane-tracker-mcp' in line]
                if status and 'Up' in status[0]:
                    print(f"{GREEN}✅ Hurricane Tracker MCP container running{RESET}")
                    print("   Container: hurricane-tracker-mcp")
                    print(f"   URL: {hurricane_url}")
                else:
                    print(f"{YELLOW}⚠️  Hurricane Tracker MCP container exists but not running{RESET}")
                    all_good = False
            else:
                print(f"{YELLOW}⚠️  Hurricane Tracker MCP container not found{RESET}")
                print("   Run: docker-compose up -d")
                all_good = False

            # Check Weather AI API container (Level 1+)
            if "weather-ai-api" in containers:
                status = [line for line in containers.split('\n') if 'weather-ai-api' in line]
                if status and 'Up' in status[0]:
                    print(f"{GREEN}✅ Weather AI API container running{RESET}")
                    print("   Container: weather-ai-api")
                    print(f"   URL: {api_url}")
                else:
                    print(f"{YELLOW}⚠️  Weather AI API container exists but not running{RESET}")
                    all_good = False
            else:
                print(f"{YELLOW}⚠️  Weather AI API container not found (needed for Level 1+){RESET}")
                print("   Run: docker-compose up -d")
                all_good = False
        else:
            print(f"{YELLOW}⚠️  Could not check Docker containers{RESET}")
            print("   Make sure Docker is running")
            all_good = False

    except Exception as e:
        print(f"{YELLOW}⚠️  Docker check failed:{RESET} {str(e)[:100]}")
        print("   Run: docker-compose up -d")
        all_good = False

    # Try to hit health endpoints
    try:
        import requests

        print("\n🔎 Checking MCP health endpoints...")

        # Weather MCP health check
        try:
            response = requests.get(f"{weather_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"{GREEN}✅ Weather MCP Server health check passed{RESET}")
            else:
                print(f"{YELLOW}⚠️  Weather MCP Server returned status {response.status_code}{RESET}")
        except requests.exceptions.RequestException:
            print(f"{YELLOW}⚠️  Weather MCP Server health endpoint not accessible{RESET}")
            print("   Make sure containers are running: docker-compose ps")

        # Hurricane MCP health check
        try:
            response = requests.get(f"{hurricane_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"{GREEN}✅ Hurricane Tracker MCP health check passed{RESET}")
            else:
                print(f"{YELLOW}⚠️  Hurricane Tracker MCP returned status {response.status_code}{RESET}")
        except requests.exceptions.RequestException:
            print(f"{YELLOW}⚠️  Hurricane Tracker MCP health endpoint not accessible{RESET}")
            print("   Make sure containers are running: docker-compose ps")

        # Weather AI API health check (Level 1+)
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"{GREEN}✅ Weather AI API health check passed{RESET}")
            else:
                print(f"{YELLOW}⚠️  Weather AI API returned status {response.status_code}{RESET}")
        except requests.exceptions.RequestException:
            print(f"{YELLOW}⚠️  Weather AI API health endpoint not accessible (needed for Level 1+){RESET}")
            print("   Make sure containers are running: docker-compose ps")

    except ImportError:
        print(f"{YELLOW}⚠️  requests library not installed, skipping health checks{RESET}")

    return all_good

def check_langsmith():
    """Verify LangSmith tracing is enabled"""
    print("\n🔎 Checking LangSmith configuration...")

    tracing = os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT", "weather-ai-agent-service")  # Default value
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if tracing == "true" and api_key:
        print(f"{GREEN}✅ LangSmith tracing enabled{RESET}")
        print(f"   Project: {project}")
        print(f"   Endpoint: {endpoint}")
        return True
    else:
        print(f"{RED}❌ LangSmith tracing not properly configured{RESET}")
        if tracing != "true":
            print(f"   LANGCHAIN_TRACING_V2={tracing} (should be 'true')")
        if not api_key:
            print("   LANGCHAIN_API_KEY is missing (required)")
        print(f"{YELLOW}   Get API key at: https://smith.langchain.com/{RESET}")
        return False

def check_docker():
    """Verify Docker is installed"""
    print("\n🔎 Checking Docker installation...")

    try:
        import subprocess

        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"{GREEN}✅ Docker is installed{RESET}")
            print(f"   {version}")

            # Check if Docker is running (optional for Level 0)
            test_result = subprocess.run(
                ["docker", "ps"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if test_result.returncode == 0:
                print(f"{GREEN}✅ Docker is running{RESET}")
            else:
                print(f"{YELLOW}⚠️  Docker installed but not running (start when needed for Level 1+){RESET}")

            return True
        else:
            print(f"{RED}❌ Docker command failed{RESET}")
            return False
    except FileNotFoundError:
        print(f"{RED}❌ Docker not found{RESET}")
        print(f"{YELLOW}   Install Docker Desktop: https://www.docker.com/products/docker-desktop/{RESET}")
        return False
    except Exception as e:
        print(f"{YELLOW}⚠️  Docker check failed:{RESET} {str(e)[:100]}")
        return False

def check_project_structure():
    """Verify project structure"""
    print("\n🔎 Checking project structure...")

    required_files = {
        "pyproject.toml": "Project configuration",
        "uv.lock": "Dependency lockfile",
        ".env.template": "Environment template",
        ".gitignore": "Git ignore rules",
        "README.md": "Project documentation",
        "CHANGELOG.md": "Version history",
    }

    all_present = True
    for file, description in required_files.items():
        if Path(file).exists():
            print(f"{GREEN}✅ {file}{RESET} - {description}")
        else:
            print(f"{RED}❌ {file}{RESET} missing - {description}")
            all_present = False

    return all_present

def check_langgraph_cli():
    """Verify LangGraph CLI is installed (for LangSmith Studio)"""
    print("\n🔎 Checking LangGraph CLI installation...")

    try:
        import subprocess

        result = subprocess.run(
            ["langgraph", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"{GREEN}✅ LangGraph CLI is installed{RESET}")
            print(f"   {version}")
            return True
        else:
            print(f"{YELLOW}⚠️  LangGraph CLI found but version check failed{RESET}")
            print(f"{YELLOW}   Reinstall: uv pip install 'langgraph-cli[inmem]'{RESET}")
            return False
    except FileNotFoundError:
        print(f"{YELLOW}⚠️  LangGraph CLI not found (needed for LangSmith Studio){RESET}")
        print(f"{YELLOW}   Install: uv pip install 'langgraph-cli[inmem]'{RESET}")
        return False
    except Exception as e:
        print(f"{YELLOW}⚠️  LangGraph CLI check failed:{RESET} {str(e)[:100]}")
        return False

def main():
    """Run all verification checks"""
    print_header("🔍 Level 1 Setup Verification (ReAct Agent + HITL)")

    checks = [
        ("Python 3.13+", check_python_version),
        ("Environment Variables", check_env_vars),
        ("Project Structure", check_project_structure),
        ("OpenAI API", check_openai_connection),
        ("Anthropic API (Optional)", check_anthropic_connection),
        ("All Docker Containers", check_mcp_servers),
        ("LangSmith", check_langsmith),
        ("Docker", check_docker),
        ("LangGraph CLI (Optional)", check_langgraph_cli),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"{RED}❌ {name} check crashed:{RESET} {str(e)[:100]}")
            results.append(False)

    print_header("📊 Verification Summary")

    passed = sum(results)
    total = len(results)

    if all(results):
        print(f"{GREEN}🎉 ALL {total} CHECKS PASSED! Level 1 environment ready!{RESET}")
        print(f"\n{GREEN}✅ Level 1 Complete:{RESET} ReAct Agent + HITL + MCP Integration")
        print(f"\n{BLUE}Next steps:{RESET}")
        print("  • Test agent: langgraph dev (opens LangSmith Studio)")
        print("  • Test API: curl http://localhost:8000/health")
        print("  • View traces: https://smith.langchain.com/")
        print("  • Start Level 2: Plan CoT + RAG implementation")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  {passed}/{total} CHECKS PASSED ({total - passed} failed){RESET}")
        print(f"\n{RED}Please fix errors above before proceeding.{RESET}")
        print(f"\n{BLUE}Common fixes:{RESET}")
        print("  • Copy .env.template to .env and add your API keys")
        print("  • Get OpenAI API key: https://platform.openai.com/api-keys")
        print("  • Get LangSmith API key: https://smith.langchain.com/")
        print("  • Start Docker containers: docker-compose up -d")
        print("  • Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        print("  • Install LangGraph CLI: uv pip install 'langgraph-cli[inmem]'")
        return 1

if __name__ == "__main__":
    exit(main())
