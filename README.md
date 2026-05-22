# Agent Trajectory Optimization

Optimizing how a small LLM behaves inside an agentic loop by fine-tuning on expert agent trajectories.

---

## Background: NL2SQL Is "Solved" Until It Isn't

Natural language to SQL is one of the most common agent use cases. The pattern looks simple: user asks a question, agent writes SQL, database returns results, agent summarizes. With a capable model behind it, this works remarkably well — demos look magical.

But in production, you hit a wall. Real databases don't look like textbook examples:

- **Schema scale**: Not 5 tables with obvious names. Try 90 tables, some with 77+ columns, where the revenue data lives in `ArItem` (Accounts Receivable Item), not anything called "revenue" or "billing."
- **Naming inconsistencies**: `Admission` (77 columns, PascalCase) coexists with `admissions` (19 columns, snake_case). Same concept, completely different structures. Which one has the data you need?
- **Hidden relationships**: Bed occupancy requires joining `BedTransfer` → `Bed` → `Class`. Patient financials require `ArItem` → `Admission` → `PatientOrganization`. No column is named helpfully enough to guess this.
- **Column ambiguity**: Five patient-related tables (`Patient`, `PatientOrganization`, `PatientBalance`, `PatientAttribute`, `PatientStatus`) — a model can't know which has the right columns without looking.

A large model handles this by treating it as an **exploration problem** — `SHOW TABLES`, `DESCRIBE`, small test queries, then targeted JOINs using columns it actually discovered. A small model skips all of this, writes a query referencing columns it assumes exist, and fails.

### The experiment behind this repo

This work came from building a SQL agent against a hospital management database (90 tables, 450+ questions ranging from "How much revenue last month?" to "Which insurance payers bring the most patients across all facilities?"). The agent framework is [Strands Agents SDK](https://github.com/strands-agents/sdk-python) with a single tool — `sql_executor`:

```python
agent = Agent(
    system_prompt="You are a SQL expert that solves problems through incremental exploration...",
    model=model,
    tools=[sql_executor]
)
result = agent("Which doctors generate the most revenue by specialty?")
```

Strands runs the agent loop: sends messages to the model → parses tool calls → executes SQL → appends results → loops until the model responds with a final answer instead of another tool call.

### How a well-behaved agent solves this

With a large model, the agent explores methodically:

```
Step 1: SHOW DATABASES                          → discovers hospital_db
Step 2: SHOW TABLES FROM hospital_db            → sees 90 tables
Step 3: DESCRIBE hospital_db.ArItem             → finds NetAmount, AdmissionId (90 columns)
Step 4: DESCRIBE hospital_db.Admission          → finds PatientId, DoctorUserId (77 columns)
Step 5: SELECT ... FROM ArItem JOIN Admission   → uses actual discovered column names
Step 6: "Based on the data, Dr. Smith generated $125K..."
```

When a query fails (`ERROR: Unknown column 'ai.BedId'`), it reasons: *"ArItem doesn't have BedId. Let me DESCRIBE ArItem to see what columns are actually there."* It adapts and continues.

### How an unoptimized small model breaks

The same agent with a small model:

```
Step 1: SELECT p.name, SUM(b.total_amount) as revenue
        FROM patients p JOIN billing b ON p.patient_id = b.patient_id
        JOIN doctors d ON b.doctor_id = d.id
        GROUP BY d.specialty
        
→ ERROR: Table 'hospital_db.billing' doesn't exist
```

There is no `billing` table — the model hallucinated it based on what hospital databases "probably" look like. The actual table is `ArItem`. The patient column is `PatientId` not `patient_id`. The doctor reference is `DoctorUserId` not `doctor_id`.

After the error:
- Retries with `bills` or `invoices` (still hallucinated)
- Switches to `admissions` (the wrong 19-column table lacking financial data)
- Or gives up: "I cannot determine this without schema information"

Even when it survives initial steps, it fails deeper. It writes a JOIN referencing `ArItem.BedId` — a column that does not exist in a 90-column table. It cannot know this without running `DESCRIBE` first, and it never does.

### The core gap

The small model knows *how* to make tool calls. It produces valid `<tool_call>` syntax. What it lacks is the **strategic reasoning** that makes tool calls effective:

- **Explore before assuming** — the discipline to DESCRIBE tables before writing queries
- **Connect observations** — reasoning like "I see BedTransfer has BedId, Bed has ClassId, so I can get occupancy per bed class"
- **Recover from errors** — interpreting "Unknown column" as "I need to check the actual schema" rather than guessing a different wrong name
- **Know when to stop** — recognizing sufficient data vs. continuing to explore irrelevant tables

This isn't a knowledge problem or a format problem. It's a **behavioral policy problem** — the model needs to learn a different decision-making strategy inside the agent loop.

### Why this generalizes

The same pattern breaks small-model agents anywhere the environment is too complex to guess:
- Code agents that assume file paths or function signatures without checking
- API agents that guess endpoint parameters instead of reading schemas
- Data pipeline agents that write transformations without inspecting actual data shapes

---

## Why This Is Different From Standard LLM Fine-tuning

Standard fine-tuning: single-turn **input→output**. Agent execution: **multi-step decision loop**.

```
Standard fine-tuning:           Agent execution:
                                
  input → output                  input → think → act → observe
                                          ↑                  │
                                          └──────────────────┘
                                          (loop N times, then answer)
```

What makes agent training fundamentally harder:

1. **Each step depends on all previous steps.** The model reasons about accumulated context — what tools returned, what failed, what's been learned.

2. **The model decides WHAT to do, not just what to say.** Call a tool (which one, what arguments) or stop and answer. A policy decision, not text generation.

3. **The context grows with each step.** By step 6 the model sees everything. It must reason over all prior results to decide the next action.

4. **Errors are part of the loop.** Failed tool calls require interpreting the error and adjusting — not just producing the next token.

5. **Knowing when to stop is critical.** Over-exploration wastes tokens and cost. Under-exploration gives wrong answers.

Fine-tuning on (question, final_answer) pairs teaches what good answers look like, not how to *arrive* at them through multi-step tool use. The model needs to learn the **intermediate decisions** — what to do at step 1, what to do at step 4 given what steps 1-3 revealed, when to stop at step 7.

---

## The Approach

```
┌──────────────────┐        ┌─────────────┐        ┌──────────────┐        ┌──────────────────────┐
│  Agent with      │ record │  Captured   │ convert│  Incremental │  fine- │  Agent with          │
│  Large Model     │───────▶│  Trajectory │───────▶│  Training    │──tune─▶│  Small Model         │
│  (expert)        │  every │  Traces     │  to    │  Data        │        │  (optimized)         │
│                  │  step  │             │per-step│              │        │                      │
│  Thinks well,    │        │  Full agent │examples│  What the    │        │  Same tools & loop,  │
│  explores,       │        │  sessions   │        │  model sees  │        │  but now makes the   │
│  recovers from   │        │  with all   │        │  at each     │        │  same strategic      │
│  errors          │        │  reasoning  │        │  decision    │        │  decisions as the    │
│                  │        │             │        │  point       │        │  expert              │
└──────────────────┘        └─────────────┘        └──────────────┘        └──────────────────────┘
```

---

## Step 1: Capture Trajectories

Run an expert model as a tool-calling agent and record every step. Most agent frameworks provide lifecycle hooks that fire during execution — register a listener that records each event.

**Relevant lifecycle events (Strands SDK example):**

| Event | Fires when | Records |
|-------|-----------|---------|
| `BeforeInvocationEvent` | Agent loop starts | System prompt, available tools |
| `MessageAddedEvent` | Message enters history | User, assistant, or tool messages |
| `AfterToolCallEvent` | Tool finishes | Tool name, arguments, result |
| `AfterInvocationEvent` | Agent loop ends | Final result, stop reason |

The output is a flat chronological list of steps. See [`sample_raw_trajectory.json`](sample_raw_trajectory.json) for a complete example — one agent session answering "Can we get a breakdown of service utilization by sales item category?"

The trajectory records:
- The system prompt and tool definitions
- The user's question
- Each assistant reasoning + tool call
- Each tool result
- The final answer

---

## Step 2: Convert to Incremental Training Data

One trajectory with N steps becomes **N training examples**. Each example replicates what the agent framework would send to the model at that point in the loop.

See [`sample_training_data.jsonl`](sample_training_data.jsonl) — the 6 incremental examples built from the same raw trajectory above.

### The Key Insight

At inference, the framework sends the **full conversation history** to the model at each step. By step 5, the model sees: system prompt + user question + all prior tool calls + all results. Training data must match this structure exactly.

### The Stripping Rule

Prior assistant messages are stripped to **just tool calls** — no reasoning. At inference the framework only stores what action was taken and what it returned, not the model's prior thinking. Training matches this:

```
Example 1: [system, user]                                        → think + tool_call
Example 3: [system, user, tool_call₁, result₁, tool_call₂, ...] → think + tool_call
Example 6: [system, user, <full history>]                        → final answer
```

Only the LAST assistant message retains `reasoning_content`. Prior ones are just `{"role": "assistant", "tool_calls": [...]}`.

### Training Data Format

Each JSONL line:

```json
{
  "messages": [
    {"role": "system", "content": "You are a SQL expert..."},
    {"role": "user", "content": [{"text": "question", "type": "text"}]},
    {"role": "assistant", "tool_calls": [{"function": {"name": "sql_executor", "arguments": "..."}}]},
    {"role": "tool", "tool_call_id": "...", "content": "[results]"},
    {"role": "assistant", "reasoning_content": "Based on the schema...", "content": "\n\n", "tool_calls": [...]}
  ]
}
```

- `reasoning_content` → maps to `<think>...</think>` tokens (Qwen's native format)
- `tool_calls` → maps to `<tool_call>...</tool_call>` tokens
- The model's chat template handles the conversion automatically

---

## Step 3: Fine-tune with Loss Masking

See [`finetune.py`](finetune.py)

The model sees the full conversation as context but only trains on the **final assistant message** in each example:

```
[system + user + prior tool_calls + tool results]          ← MASKED (labels = -100)
[<think>reasoning</think> <tool_call>action</tool_call>]   ← TRAINED (gradient flows)
```

This teaches the decision policy — "given this accumulated state, what should I think and do next?" — without wasting gradient on reproducing context the framework already provides.

---

## What Gets Optimized

Not knowledge. Not tool-calling mechanics. The **behavioral policy** inside the agent loop:

| Behavior | Base model | After trajectory optimization |
|----------|-----------|-------------------------------|
| First action | Tries complex query immediately | Explores schema first |
| Mid-loop reasoning | Generic "let me query" | "I see BedTransfer has admission_id, so I can JOIN..." |
| Error handling | Retries same failing query | Reasons about error, adjusts approach |
| Termination | Over-explores or stops too early | Recognizes sufficient data, stops |
| Answer format | Verbose process explanation | Direct "Based on the data, ..." |

---

## Serving

After fine-tuning, the model plugs into any agent framework as a drop-in backend. The framework doesn't know the model was trajectory-optimized — it just sends messages and gets back tool calls or content.

---

## Files

| File | What it is |
|------|-----------|
| `sample_raw_trajectory.json` | One full captured agent trace (input to step 2) |
| `sample_training_data.jsonl` | 6 incremental examples built from that trace (input to step 3) |
| `finetune.py` | Training script with loss masking |

## Notes

- Teacher model must have a license permitting use of outputs for training
- Qwen3 used as student — natively supports `<think>` and `<tool_call>` tokens
- Context length matters: agent conversations need 16k-49k tokens
- One epoch sufficient; ~450 trajectories (~4000 examples) showed clear behavioral change
- The approach applies to any tool-calling agent domain, not just SQL
