# Agent Trajectory Optimization

Optimizing how a small LLM behaves inside an agentic loop by fine-tuning on expert agent trajectories.

## Why This Is Different From Standard LLM Fine-tuning

Standard LLM fine-tuning operates on **single-turn input→output pairs**. You give the model a prompt, it produces a response, done.

Agent execution is fundamentally different:

```
Standard fine-tuning:           Agent execution:
                                
  input → output                  input → think → act → observe
                                          ↑                  │
                                          └──────────────────┘
                                          (loop N times, then answer)
```

An agent operates in a **multi-step decision loop** where:

1. **Each step depends on all previous steps.** The model reasons about accumulated context — what tools returned, what failed, what's been learned so far.

2. **The model decides WHAT to do, not just what to say.** At each step it chooses between calling a tool (which one, with what arguments) or stopping and answering. This is a policy decision, not text generation.

3. **The context grows with each step.** By step 6, the model sees full history of prior tool calls and results. It must reason over all of this to decide the next action.

4. **Errors are part of the loop.** When a tool call fails, the model must interpret the error, adjust its approach, and try something different.

5. **Knowing when to stop is critical.** Over-exploration wastes tokens and money. Under-exploration gives wrong answers. The termination decision is as important as any tool call.

### The Problem With Small Models as Agents

Small models (4B-30B) know the *mechanics* of tool calling — they produce `<tool_call>` tokens in the right format. What they lack is **strategic behavior**:

- They try to write the perfect query immediately (skipping exploration → fails)
- They don't connect results from step 3 to decisions at step 5
- They don't know when they have enough information to stop
- They don't recover gracefully from errors

A large model (70B+) does this naturally. The goal: transfer that behavioral policy into the small model.

### Why Standard Fine-tuning Doesn't Solve This

Fine-tuning on (question, final_answer) pairs teaches what good answers look like, not how to *arrive* at them through a tool-calling loop. The model needs to learn the **intermediate decision-making** — what to do at step 1, what to do at step 4 given what steps 1-3 revealed, and when to stop at step 7.

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
