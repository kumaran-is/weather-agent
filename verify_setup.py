"""
Level 0 Setup Verification Script
Ensures all components are properly configured before Level 1.

Tests:
1. Python 3.13+ installed
2. Environment variables set (Docker + HTTP transport)
3. OpenAI API connection
4. Anthropic API connection (optional)
5. MCP Docker containers running (Weather + Hurricane)
6. MCP health endpoints accessible
7. LangSmith tracing enabled
8. Docker installed and running
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
    print(f"🔎 Checking Python version...")
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
    print(f"\n🔎 Checking environment variables...")

    required_vars = {
        "OPENAI_API_KEY": False,  # Required
        "LANGCHAIN_API_KEY": False,  # Required
        "LANGCHAIN_TRACING_V2": False,  # Required
        "MCP_WEATHER_SERVER_URL": False,  # Required (Docker + HTTP)
        "MCP_HURRICANE_SERVER_URL": False,  # Required (Docker + HTTP)
    }

    optional_vars = {
        "ANTHROPIC_API_KEY": "Optional - for Claude models",
        "MCP_WEATHER_SERVER_ENABLED": "Optional - defaults to true",
        "MCP_HURRICANE_SERVER_ENABLED": "Optional - defaults to true"
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
    print(f"\n🔎 Checking OpenAI API connection...")

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
    print(f"\n🔎 Checking Anthropic API connection (optional)...")

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
    print(f"\n🔎 Checking MCP Docker containers...")

    import subprocess

    weather_url = os.getenv("MCP_WEATHER_SERVER_URL", "http://localhost:8080")
    hurricane_url = os.getenv("MCP_HURRICANE_SERVER_URL", "http://localhost:8081")

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
                    print(f"   Container: weather-mcp-server")
                    print(f"   URL: {weather_url}")
                else:
                    print(f"{YELLOW}⚠️  Weather MCP Server container exists but not running{RESET}")
                    all_good = False
            else:
                print(f"{YELLOW}⚠️  Weather MCP Server container not found{RESET}")
                print(f"   Run: docker-compose up -d")
                all_good = False

            # Check Hurricane MCP container
            if "hurricane-tracker-mcp" in containers:
                status = [line for line in containers.split('\n') if 'hurricane-tracker-mcp' in line]
                if status and 'Up' in status[0]:
                    print(f"{GREEN}✅ Hurricane Tracker MCP container running{RESET}")
                    print(f"   Container: hurricane-tracker-mcp")
                    print(f"   URL: {hurricane_url}")
                else:
                    print(f"{YELLOW}⚠️  Hurricane Tracker MCP container exists but not running{RESET}")
                    all_good = False
            else:
                print(f"{YELLOW}⚠️  Hurricane Tracker MCP container not found{RESET}")
                print(f"   Run: docker-compose up -d")
                all_good = False
        else:
            print(f"{YELLOW}⚠️  Could not check Docker containers{RESET}")
            print(f"   Make sure Docker is running")
            all_good = False

    except Exception as e:
        print(f"{YELLOW}⚠️  Docker check failed:{RESET} {str(e)[:100]}")
        print(f"   Run: docker-compose up -d")
        all_good = False

    # Try to hit health endpoints
    try:
        import requests

        print(f"\n🔎 Checking MCP health endpoints...")

        # Weather MCP health check
        try:
            response = requests.get(f"{weather_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"{GREEN}✅ Weather MCP Server health check passed{RESET}")
            else:
                print(f"{YELLOW}⚠️  Weather MCP Server returned status {response.status_code}{RESET}")
        except requests.exceptions.RequestException:
            print(f"{YELLOW}⚠️  Weather MCP Server health endpoint not accessible{RESET}")
            print(f"   Make sure containers are running: docker-compose ps")

        # Hurricane MCP health check
        try:
            response = requests.get(f"{hurricane_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"{GREEN}✅ Hurricane Tracker MCP health check passed{RESET}")
            else:
                print(f"{YELLOW}⚠️  Hurricane Tracker MCP returned status {response.status_code}{RESET}")
        except requests.exceptions.RequestException:
            print(f"{YELLOW}⚠️  Hurricane Tracker MCP health endpoint not accessible{RESET}")
            print(f"   Make sure containers are running: docker-compose ps")

    except ImportError:
        print(f"{YELLOW}⚠️  requests library not installed, skipping health checks{RESET}")

    return all_good

def check_langsmith():
    """Verify LangSmith tracing is enabled"""
    print(f"\n🔎 Checking LangSmith configuration...")

    tracing = os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGCHAIN_API_KEY")
    project = os.getenv("LANGCHAIN_PROJECT")

    if tracing == "true" and api_key and project:
        print(f"{GREEN}✅ LangSmith tracing enabled{RESET}")
        print(f"   Project: {project}")
        return True
    else:
        print(f"{RED}❌ LangSmith tracing not properly configured{RESET}")
        if tracing != "true":
            print(f"   LANGCHAIN_TRACING_V2={tracing} (should be 'true')")
        if not api_key:
            print(f"   LANGCHAIN_API_KEY is missing")
        if not project:
            print(f"   LANGCHAIN_PROJECT is missing")
        print(f"{YELLOW}   Get API key at: https://smith.langchain.com/{RESET}")
        return False

def check_docker():
    """Verify Docker is installed"""
    print(f"\n🔎 Checking Docker installation...")

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
    print(f"\n🔎 Checking project structure...")

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

def main():
    """Run all verification checks"""
    print_header("🔍 Level 0 Setup Verification")

    checks = [
        ("Python 3.13+", check_python_version),
        ("Environment Variables", check_env_vars),
        ("Project Structure", check_project_structure),
        ("OpenAI API", check_openai_connection),
        ("Anthropic API (Optional)", check_anthropic_connection),
        ("MCP Docker Containers", check_mcp_servers),
        ("LangSmith", check_langsmith),
        ("Docker", check_docker),
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
        print(f"{GREEN}🎉 ALL {total} CHECKS PASSED! You're ready for Level 1!{RESET}")
        print(f"\n{GREEN}Next step:{RESET} Proceed to Level 1 implementation")
        print(f"  git checkout -b level-1-react-agent-hitl")
        print(f"\n{BLUE}Reminder:{RESET} If you haven't already:")
        print(f"  1. Copy .env.template to .env")
        print(f"  2. Add your API keys to .env")
        print(f"  3. Start Docker Desktop (when needed for Level 1+)")
        return 0
    else:
        print(f"\n{YELLOW}⚠️  {passed}/{total} CHECKS PASSED ({total - passed} failed){RESET}")
        print(f"\n{RED}Please fix errors above before proceeding to Level 1.{RESET}")
        print(f"\n{BLUE}Common fixes:{RESET}")
        print(f"  • Copy .env.template to .env and add your API keys")
        print(f"  • Get OpenAI API key: https://platform.openai.com/api-keys")
        print(f"  • Get LangSmith API key: https://smith.langchain.com/")
        print(f"  • Install Docker Desktop: https://www.docker.com/products/docker-desktop/")
        return 1

if __name__ == "__main__":
    exit(main())
