# Neo4j Desktop Client Setup

Quick guide to connect Neo4j Desktop to the Weather AI graph database.

**← Back to [Main README](../../README.md)**

---

## Table of Contents

1. [Quick Reference](#quick-reference)
2. [Install Neo4j Desktop](#install-neo4j-desktop)
3. [Connect to Weather AI Neo4j](#connect-to-weather-ai-neo4j)
4. [Explore Graphiti Data](#explore-graphiti-data)
5. [Run Queries](#run-queries)
6. [Troubleshooting](#troubleshooting)

---

## Quick Reference

**Connection Settings** (Copy-Paste Ready):
```
Protocol:       bolt://
Connection URL: localhost:7687          ⚠️ Use localhost, NOT neo4j
Database user:  neo4j
Password:       weatherai2025
```

**URLs**:
- Neo4j Browser: http://localhost:7474
- Bolt Protocol: bolt://localhost:7687

**Container**:
- Name: `weather-ai-neo4j-dev`
- Image: `neo4j:5.26.0`
- Database: `weather_ai` (Graphiti storage)

**Common Mistakes**:
- ❌ Using `neo4j:7687` (Docker hostname) → Use `localhost:7687`
- ❌ Wrong password → Must be `weatherai2025`
- ❌ Container not running → Run `make docker-up-dev`

---

## Install Neo4j Desktop

### Download

| Platform | Download Link |
|----------|--------------|
| **macOS** | https://neo4j.com/download/ |
| **Windows** | https://neo4j.com/download/ |
| **Linux** | https://neo4j.com/download/ |

### Install
1. Download Neo4j Desktop from the link above
2. Create a free Neo4j account (or skip to use offline)
3. Open the installer
4. Follow installation prompts
5. Launch Neo4j Desktop

---

## Connect to Weather AI Neo4j

### Prerequisites
- Weather AI services running: `docker-compose up -d`
- Neo4j container healthy: `docker ps | grep neo4j`

### Connection Steps

1. **Open Neo4j Desktop** → Click **New** → **Remote Connection**

2. **Enter Connection Details**:
   ```
   Name:          Weather AI Graph
   Protocol:      bolt://
   Connection URL: localhost:7687
   Database user: neo4j
   Password:      weatherai2025
   ```

   **⚠️ IMPORTANT**: Use `localhost:7687` NOT `neo4j:7687`
   - `localhost` = Connect from your host machine (Neo4j Desktop)
   - `neo4j` = Docker container hostname (only works inside Docker network)

3. **Click "Connect"**

4. **Open Neo4j Browser** → Click "Open" on the remote connection

5. **Connection Details** (reference):
   | Field | Value | Notes |
   |-------|-------|-------|
   | **Protocol** | `bolt://` | Bolt protocol for Neo4j |
   | **Connection URL** | `localhost:7687` | Use localhost, NOT neo4j |
   | **Browser URL** | `http://localhost:7474` | Web interface |
   | **Database user** | `neo4j` | Default username |
   | **Password** | `weatherai2025` | From docker-compose.dev.yml:213 |
   | **Database** | `weather_ai` | Graphiti database name |
   | **Neo4j Version** | `5.26.0` | Community Edition |
   | **Container Name** | `weather-ai-neo4j-dev` | Development environment |

---

## Explore Graphiti Data

### View All Nodes

**Show all nodes and relationships (limited to 25 for performance)**:
```cypher
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 25
```

**Show all nodes (any type)**:
```cypher
MATCH (n)
RETURN n
LIMIT 50
```

### Graphiti Node Types (Actual Schema)

Your Weather AI Graphiti database contains these node types:

| Node Type | Count | Description |
|-----------|-------|-------------|
| **Entity** | 19 | Extracted entities and facts from conversations |
| **Episodic** | 9 | Episode/conversation memory nodes |
| **Community** | 0 | Graph community detection (clustering) |

**Relationship Types**:
- `RELATES_TO` - Connects related entities and episodes
- `MENTIONS` - Links entities to episodes where mentioned
- `HAS_MEMBER` - Community membership (graph clustering)

### View All Entities

**Show all entity nodes**:
```cypher
MATCH (e:Entity)
RETURN e
LIMIT 20
```

**Show entities with their properties**:
```cypher
MATCH (e:Entity)
RETURN e.name, e.uuid, properties(e) AS all_properties
LIMIT 10
```

### View Episodic Memories

**Show all episodic memory nodes**:
```cypher
MATCH (ep:Episodic)
RETURN ep
LIMIT 20
```

**Show episodes with their timestamps**:
```cypher
MATCH (ep:Episodic)
RETURN ep.name, ep.created_at, ep.content
ORDER BY ep.created_at DESC
LIMIT 10
```

### View Entity Relationships

**Show which entities are mentioned in episodes**:
```cypher
MATCH (e:Entity)-[r:MENTIONS]->(ep:Episodic)
RETURN e.name AS Entity, ep.name AS Episode, type(r) AS Relationship
LIMIT 10
```

**Show related entities**:
```cypher
MATCH (e1:Entity)-[r:RELATES_TO]->(e2:Entity)
RETURN e1.name AS Entity1, e2.name AS Entity2, type(r) AS Relationship
LIMIT 10
```

**Note**: Run each query separately. Don't copy-paste multiple queries together or you'll get: `RETURN can only be used at the end of the query`

---

## Run Queries

### Example Queries (Using Actual Graphiti Schema)

**1. Find entities by name pattern**:
```cypher
MATCH (e:Entity)
WHERE e.name CONTAINS "user"
RETURN e.name, e.summary, e.group_id
LIMIT 10
```

**2. Get recent episodic memories**:
```cypher
MATCH (ep:Episodic)
WHERE ep.created_at > datetime("2025-12-07T00:00:00Z")
RETURN ep.name, ep.content, ep.created_at
ORDER BY ep.created_at DESC
LIMIT 10
```

**3. Find entities by group**:
```cypher
MATCH (e:Entity)
WHERE e.group_id = "user_profiles"
RETURN e.name, e.summary
LIMIT 10
```

**4. Count nodes by type**:
```cypher
MATCH (n)
RETURN labels(n)[0] AS NodeType, count(*) AS Count
ORDER BY Count DESC
```

**5. Visualize entity-episode graph**:
```cypher
MATCH (e:Entity)-[r]-(ep:Episodic)
RETURN e, r, ep
LIMIT 25
```

**6. Find related entities**:
```cypher
MATCH (e1:Entity)-[r:RELATES_TO]-(e2:Entity)
RETURN e1.name AS Entity1, type(r) AS Relationship, e2.name AS Entity2
LIMIT 10
```

---

## Verify Connection

### Check Database Status

**Show database info**:
```cypher
CALL dbms.components() YIELD name, versions, edition
RETURN name, versions, edition
```

**Expected Output**:
```
name: "Neo4j Kernel"
versions: ["5.26.0"]
edition: "community"
```

### Check Graphiti Schema

**Show all node labels**:
```cypher
CALL db.labels()
```

**Actual Labels** (current database):
- `Entity` (19 nodes) - Graphiti entity nodes (facts extracted from conversations)
- `Episodic` (9 nodes) - Graphiti episode nodes (conversation memories)
- `Community` (0 nodes) - Graph community detection for clustering

**Relationship Types**:
- `RELATES_TO` - Connects related entities and episodes
- `MENTIONS` - Links episodes to entities mentioned
- `HAS_MEMBER` - Community membership relationships

**Count nodes by type**:
```cypher
MATCH (n)
RETURN labels(n)[0] AS NodeType, count(*) AS Count
ORDER BY Count DESC
```

---
