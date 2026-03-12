import sqlite3
import math
import os
import sys
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DB_PATH = "smollerlm.db"
MODEL_NAME = "mehmetkeremturkcan/SmollerLM2-10M-sftb"

def build_optimized_sql_db():
    if os.path.exists(DB_PATH): os.remove(DB_PATH)
    
    print("Downloading Model and Tokenizer...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    sd = model.state_dict()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.executescript("""
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = 0;
        PRAGMA page_size = 32768;
    """)
    
    d60_cols = ", ".join(f"d{i} REAL" for i in range(60))
    d60_qs = ", ".join("?" for _ in range(60))
    
    tables = [
        f"w_emb (token INT PRIMARY KEY, {d60_cols})",
        f"w_norm (layer INT PRIMARY KEY, {d60_cols})",
        f"w_q (layer INT, head INT, h_dim INT, {d60_cols})",
        f"w_k (layer INT, head INT, h_dim INT, {d60_cols})",
        f"w_v (layer INT, head INT, h_dim INT, {d60_cols})",
        f"w_o (layer INT, in_idx INT, {d60_cols})",
        f"w_ffn_norm (layer INT PRIMARY KEY, {d60_cols})",
        f"w_ffn_gate (layer INT, out_dim INT, {d60_cols})",
        f"w_ffn_up (layer INT, out_dim INT, {d60_cols})",
        f"w_ffn_down (layer INT, in_idx INT, {d60_cols})",
        f"w_out_norm (id INT PRIMARY KEY, {d60_cols})",
        f"w_out (token INT PRIMARY KEY, {d60_cols})",
        "rope (pos INT, h_dim INT, cos_val REAL, sin_val REAL, PRIMARY KEY(pos, h_dim)) WITHOUT ROWID",
        
        "hyperparams (id INT PRIMARY KEY, temp REAL, top_p REAL, top_k INT, min_p REAL, rep_pen REAL, max_pos INT)",
        "vocab (token INT PRIMARY KEY, text TEXT)",
        "prompt_tokens (pos INT PRIMARY KEY, token INT)",
        "generation_loop (pos INT PRIMARY KEY, token INT)",
        
        f"t_x ({d60_cols})",
        f"t_x_norm ({d60_cols})",
        "t_q (head INT, h_dim INT, val REAL, PRIMARY KEY(head, h_dim)) WITHOUT ROWID",
        "t_k (head INT, h_dim INT, val REAL, PRIMARY KEY(head, h_dim)) WITHOUT ROWID",
        "t_v (head INT, h_dim INT, val REAL, PRIMARY KEY(head, h_dim)) WITHOUT ROWID",
        "t_q_rope (head INT, h_dim INT, val REAL, PRIMARY KEY(head, h_dim)) WITHOUT ROWID",
        "t_k_rope (head INT, h_dim INT, val REAL, PRIMARY KEY(head, h_dim)) WITHOUT ROWID",
        "kv_cache (layer INT, head INT, pos INT, h_dim INT, k_val REAL, v_val REAL, PRIMARY KEY(layer, head, pos, h_dim)) WITHOUT ROWID",
        "t_attn_out (head INT, h_dim INT, val REAL, PRIMARY KEY(head, h_dim)) WITHOUT ROWID",
        "t_silu (out_dim INT, val REAL, PRIMARY KEY(out_dim)) WITHOUT ROWID",
        "t_logits (token INT, val REAL, PRIMARY KEY(token))"
    ]
    for tbl in tables: c.execute(f"CREATE TABLE IF NOT EXISTS {tbl};")

    c.execute("CREATE INDEX IF NOT EXISTS idx_kv_k ON kv_cache(layer, head, h_dim, pos);")
    c.execute("INSERT INTO hyperparams VALUES (1, 0.8, 0.9, 40, 0.1, 1.2, 2048)")

    def to_list(tensor): return tensor.to(torch.float32).flatten().tolist()
    def to_matrix(tensor): return tensor.to(torch.float32).tolist()

    print("Inserting Weights...")
    c.executemany(f"INSERT INTO w_emb VALUES (?, {d60_qs})", [[t] + row for t, row in enumerate(to_matrix(sd["model.embed_tokens.weight"]))])
    c.executemany(f"INSERT INTO w_out VALUES (?, {d60_qs})", [[t] + row for t, row in enumerate(to_matrix(sd["lm_head.weight"]))])
    c.execute(f"INSERT INTO w_out_norm VALUES (0, {d60_qs})", to_list(sd["model.norm.weight"]))

    rope_data = []
    for pos in range(2048):
        for h_dim in range(64):
            freq = 1.0 / (100000.0 ** ((h_dim // 2 * 2) / 64.0))
            rope_data.append((pos, h_dim, math.cos(pos * freq), math.sin(pos * freq)))
    c.executemany("INSERT INTO rope VALUES (?, ?, ?, ?)", rope_data)

    for i in range(30):
        prefix = f"model.layers.{i}."
        c.execute(f"INSERT INTO w_norm VALUES (?, {d60_qs})", [i] + to_list(sd[prefix+"input_layernorm.weight"]))
        c.execute(f"INSERT INTO w_ffn_norm VALUES (?, {d60_qs})", [i] + to_list(sd[prefix+"post_attention_layernorm.weight"]))
        c.executemany(f"INSERT INTO w_q VALUES (?, ?, ?, {d60_qs})", [[i, h // 64, h % 64] + r for h, r in enumerate(to_matrix(sd[prefix+"self_attn.q_proj.weight"]))])
        c.executemany(f"INSERT INTO w_k VALUES (?, ?, ?, {d60_qs})", [[i, h // 64, h % 64] + r for h, r in enumerate(to_matrix(sd[prefix+"self_attn.k_proj.weight"]))])
        c.executemany(f"INSERT INTO w_v VALUES (?, ?, ?, {d60_qs})", [[i, h // 64, h % 64] + r for h, r in enumerate(to_matrix(sd[prefix+"self_attn.v_proj.weight"]))])
        c.executemany(f"INSERT INTO w_ffn_gate VALUES (?, ?, {d60_qs})", [[i, idx] + r for idx, r in enumerate(to_matrix(sd[prefix+"mlp.gate_proj.weight"]))])
        c.executemany(f"INSERT INTO w_ffn_up VALUES (?, ?, {d60_qs})", [[i, idx] + r for idx, r in enumerate(to_matrix(sd[prefix+"mlp.up_proj.weight"]))])
        c.executemany(f"INSERT INTO w_o VALUES (?, ?, {d60_qs})", [[i, idx] + list(r) for idx, r in enumerate(list(zip(*to_matrix(sd[prefix+"self_attn.o_proj.weight"]))))])
        c.executemany(f"INSERT INTO w_ffn_down VALUES (?, ?, {d60_qs})", [[i, idx] + list(r) for idx, r in enumerate(list(zip(*to_matrix(sd[prefix+"mlp.down_proj.weight"]))))])

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    vocab_list = [(v, k.replace('Ġ', ' ').replace('Ċ', '\n')) for k, v in tok.get_vocab().items()]
    c.executemany("INSERT INTO vocab VALUES (?, ?)", vocab_list)
    
    print("Forging the High-Speed Forward-Pass Trigger...")
    d60_select = ", ".join(f"d{i}" for i in range(60))
    dot_sq = " + ".join(f"val.d{i}*val.d{i}" for i in range(60))
    rms_calc = f"1.0 / SQRT(({dot_sq}) / 60.0 + 1e-5)"
    dot_prod = " + ".join(f"w.d{i} * x.d{i}" for i in range(60))
    
    sql = [f"""
    CREATE TRIGGER IF NOT EXISTS llm_forward_pass
    AFTER INSERT ON generation_loop
    BEGIN
        DELETE FROM t_x; INSERT INTO t_x SELECT {d60_select} FROM w_emb WHERE token = NEW.token;
    """]

    for L in range(30):
        norm_calc = ", ".join(f"val.d{i} * stats.rms * wn.d{i} as d{i}" for i in range(60))
        o_sum = ", ".join(f"COALESCE(SUM(a.val * w.d{i}), 0.0) as d{i}" for i in range(60))
        d_sum = ", ".join(f"COALESCE(SUM(s.val * w.d{i}), 0.0) as d{i}" for i in range(60))
        o_update = ", ".join(f"d{i} = t_x.d{i} + COALESCE(delta.d{i}, 0.0)" for i in range(60))
        
        sql.append(f"""
        DELETE FROM t_x_norm; INSERT INTO t_x_norm SELECT {norm_calc} FROM t_x val CROSS JOIN (SELECT {rms_calc} AS rms FROM t_x val) stats JOIN w_norm wn ON wn.layer = {L};
        DELETE FROM t_q; INSERT INTO t_q SELECT w.head, w.h_dim, ({dot_prod}) FROM w_q w CROSS JOIN t_x_norm x WHERE w.layer = {L};
        DELETE FROM t_k; INSERT INTO t_k SELECT w.head, w.h_dim, ({dot_prod}) FROM w_k w CROSS JOIN t_x_norm x WHERE w.layer = {L};
        DELETE FROM t_v; INSERT INTO t_v SELECT w.head, w.h_dim, ({dot_prod}) FROM w_v w CROSS JOIN t_x_norm x WHERE w.layer = {L};
        
        DELETE FROM t_q_rope; INSERT INTO t_q_rope SELECT q.head, q.h_dim, CASE WHEN q.h_dim < 32 THEN q.val * r.cos_val - COALESCE(q_pair.val, 0.0) * r.sin_val ELSE COALESCE(q_pair.val, 0.0) * r.sin_val + q.val * r.cos_val END FROM t_q q LEFT JOIN t_q q_pair ON q.head = q_pair.head AND q_pair.h_dim = (q.h_dim + 32) % 64 JOIN rope r ON r.pos = NEW.pos AND r.h_dim = (q.h_dim % 32) * 2;
        DELETE FROM t_k_rope; INSERT INTO t_k_rope SELECT k.head, k.h_dim, CASE WHEN k.h_dim < 32 THEN k.val * r.cos_val - COALESCE(k_pair.val, 0.0) * r.sin_val ELSE COALESCE(k_pair.val, 0.0) * r.sin_val + k.val * r.cos_val END FROM t_k k LEFT JOIN t_k k_pair ON k.head = k_pair.head AND k_pair.h_dim = (k.h_dim + 32) % 64 JOIN rope r ON r.pos = NEW.pos AND r.h_dim = (k.h_dim % 32) * 2;
        
        INSERT INTO kv_cache (layer, head, pos, h_dim, k_val, v_val) SELECT {L}, k.head, NEW.pos, k.h_dim, k.val, v.val FROM t_k_rope k JOIN t_v v ON k.head = v.head AND k.h_dim = v.h_dim;

        DELETE FROM t_attn_out; INSERT INTO t_attn_out (head, h_dim, val)
        WITH 
          scores AS (SELECT q.head, kv.pos, SUM(q.val * kv.k_val) / 8.0 as val FROM t_q_rope q JOIN kv_cache kv ON kv.layer = {L} AND kv.head = CAST(q.head / 3 AS INT) AND kv.h_dim = q.h_dim WHERE kv.pos <= NEW.pos GROUP BY q.head, kv.pos),
          m_v AS (SELECT head, MAX(val) as mx FROM scores GROUP BY head),
          e_v AS (SELECT s.head, s.pos, EXP(MAX(MIN(s.val - m_v.mx, 50.0), -50.0)) as ev FROM scores s JOIN m_v ON s.head = m_v.head),
          s_v AS (SELECT head, SUM(ev) as sv FROM e_v GROUP BY head),
          softmax AS (SELECT e.head, e.pos, e.ev / s.sv as val FROM e_v e JOIN s_v s ON e.head = s.head)
        SELECT s.head, kv.h_dim, SUM(s.val * kv.v_val) FROM softmax s JOIN kv_cache kv ON kv.layer = {L} AND kv.head = CAST(s.head / 3 AS INT) AND kv.pos = s.pos GROUP BY s.head, kv.h_dim;

        UPDATE t_x SET {o_update} FROM (SELECT {o_sum} FROM t_attn_out a JOIN w_o w ON (a.head*64 + a.h_dim) = w.in_idx WHERE w.layer = {L}) AS delta;

        DELETE FROM t_x_norm; INSERT INTO t_x_norm SELECT {norm_calc} FROM t_x val CROSS JOIN (SELECT {rms_calc} AS rms FROM t_x val) stats JOIN w_ffn_norm wn ON wn.layer = {L};

        DELETE FROM t_silu; 
        INSERT INTO t_silu 
        SELECT g.out_dim, (g.val / (1.0 + EXP(MAX(MIN(-g.val, 50.0), -50.0)))) * u.val 
        FROM (SELECT w.out_dim, ({dot_prod}) as val FROM w_ffn_gate w CROSS JOIN t_x_norm x WHERE w.layer = {L}) g
        JOIN (SELECT w.out_dim, ({dot_prod}) as val FROM w_ffn_up w CROSS JOIN t_x_norm x WHERE w.layer = {L}) u
        ON g.out_dim = u.out_dim;

        UPDATE t_x SET {o_update} FROM (SELECT {d_sum} FROM t_silu s JOIN w_ffn_down w ON s.out_dim = w.in_idx WHERE w.layer = {L}) AS delta;
        """)

    final_norm_calc = ", ".join(f"val.d{i} * stats.rms * wn.d{i} as d{i}" for i in range(60))
    sql.append(f"""
        DELETE FROM t_x_norm; INSERT INTO t_x_norm SELECT {final_norm_calc} FROM t_x val CROSS JOIN (SELECT {rms_calc} AS rms FROM t_x val) stats JOIN w_out_norm wn ON wn.id = 0;
        DELETE FROM t_logits; INSERT OR REPLACE INTO t_logits SELECT w.token, ({dot_prod}) FROM w_out w CROSS JOIN t_x_norm x;
        
        UPDATE t_logits SET val = CASE WHEN val <= 0 THEN val * (SELECT rep_pen FROM hyperparams LIMIT 1) ELSE val / (SELECT rep_pen FROM hyperparams LIMIT 1) END
        WHERE token IN (SELECT token FROM generation_loop WHERE pos >= NEW.pos - 64) AND token > 16;
    END;
    """)
    
    c.executescript("\n".join(sql))
    conn.commit()
    conn.close()
    print("Database built successfully!")

def write_prompt(user_text, max_new_tokens=100):
    print(f"Tokenizing prompt and writing to {DB_PATH}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    system_prompt = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
    full_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n"
    
    tokens = tok.encode(full_text, add_special_tokens=False)
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM prompt_tokens;")
    c.executemany("INSERT INTO prompt_tokens VALUES (?, ?)", [(p, t) for p, t in enumerate(tokens)])
    
    abs_max = len(tokens) + max_new_tokens
    c.execute("UPDATE hyperparams SET max_pos = ?", (abs_max,))
    conn.commit()
    conn.close()
    
    print(f"\nPrompt injected! ({len(tokens)} context tokens). Generation is capped at {max_new_tokens} output tokens.")

def generate_and_output_sql(max_new_tokens, to_clipboard):
    sql = """CREATE VIEW IF NOT EXISTS v_next_token AS
WITH 
  hp AS (SELECT * FROM hyperparams LIMIT 1),
  next_pos AS (SELECT MAX(pos) + 1 as val FROM generation_loop),
  
  -- Check if we need to use a prompt token
  prompt_token AS (SELECT token FROM prompt_tokens WHERE pos = (SELECT val FROM next_pos)),
  
  -- If not, run the sampler
  sampled_token AS (
    SELECT token FROM (
      WITH 
       temp_logits AS (SELECT token, val / hp.temp as t_val FROM t_logits CROSS JOIN hp),
       stats AS (SELECT MAX(t_val) as max_val FROM temp_logits),
       exp_logits AS (SELECT token, t_val, EXP(MAX(MIN(t_val - stats.max_val, 50.0), -50.0)) as e_val FROM temp_logits CROSS JOIN stats),
       sum_exp AS (SELECT SUM(e_val) as s_val FROM exp_logits),
       probs AS (SELECT token, t_val, e_val / sum_exp.s_val as prob FROM exp_logits CROSS JOIN sum_exp),
       max_p_cte AS (SELECT MAX(prob) as max_p FROM probs),
       min_p_filtered AS (
           SELECT p.token, p.t_val, p.prob 
           FROM probs p CROSS JOIN max_p_cte m JOIN hp ON 1=1
           WHERE p.prob >= m.max_p * hp.min_p
       ),
       ranked_filtered AS (
           SELECT token, t_val, prob, ROW_NUMBER() OVER (ORDER BY prob DESC) as rn
           FROM min_p_filtered
       ),
       filtered AS (
           SELECT token, t_val, prob 
           FROM ranked_filtered CROSS JOIN hp 
           WHERE rn <= hp.top_k
       ),
       cumulative AS (
           SELECT token, t_val, prob, SUM(prob) OVER (ORDER BY prob DESC ROWS UNBOUNDED PRECEDING) as cdf
           FROM filtered
       ),
       top_p_filtered AS (
           SELECT token, t_val
           FROM cumulative CROSS JOIN hp
           WHERE cdf - prob <= hp.top_p
       ),
       final_selection AS (
           SELECT token
           FROM top_p_filtered
           ORDER BY (t_val - LN(-LN((ABS(RANDOM()) % 999999 + 1) / 1000000.0))) DESC
           LIMIT 1
       )
       SELECT token FROM final_selection
    )
  )
SELECT 
  (SELECT val FROM next_pos) AS pos,
  COALESCE((SELECT token FROM prompt_token), (SELECT token FROM sampled_token)) AS token
-- THIS IS THE MAGIC STOP CONDITION:
WHERE (SELECT MAX(pos) FROM generation_loop) < (SELECT max_pos FROM hyperparams)
  AND (SELECT token FROM generation_loop ORDER BY pos DESC LIMIT 1) NOT IN (0, 2);

PRAGMA journal_mode = MEMORY;
PRAGMA synchronous = 0;
PRAGMA temp_store = MEMORY;

-- 1. Prime the Database
DELETE FROM generation_loop;
DELETE FROM kv_cache;
INSERT INTO generation_loop (pos, token) SELECT pos, token FROM prompt_tokens;

"""
    sql += f"-- 2. \"Loop\" for {max_new_tokens} tokens\n"
    sql += "INSERT INTO generation_loop (pos, token) SELECT pos, token FROM v_next_token;\n" * max_new_tokens
    
    sql += """
-- 3. Print the Result
SELECT GROUP_CONCAT(v.text, '') AS Output
FROM generation_loop g
JOIN vocab v ON g.token = v.token
WHERE g.pos >= (SELECT COUNT(*) FROM prompt_tokens) 
ORDER BY g.pos;
"""

    if to_clipboard:
        copied = False
        try:
            subprocess.run("pbcopy", text=True, input=sql, check=True) # Mac
            copied = True
        except Exception:
            try:
                subprocess.run("clip", text=True, input=sql, check=True) # Windows
                copied = True
            except Exception:
                try:
                    subprocess.run(["xclip", "-selection", "clipboard"], text=True, input=sql, check=True) # Linux
                    copied = True
                except Exception:
                    pass
        
        # If successfully copied to clipboard, exit without creating a file
        if copied:
            print("\n>>> SQL run command successfully copied to clipboard! <<<")
            print(f"    Open {DB_PATH} in your DB terminal/browser and paste it.")
            return
        else:
            print("\n>>> Clipboard copy failed (no pbcopy/clip/xclip found). Writing to file instead. <<<")
    
    # Fallback/Default: Write to file
    filename = "run_inference.sql"
    with open(filename, "w") as f:
        f.write(sql)
    print(f"\n>>> SQL run command generated and saved to {filename} <<<")
    print(f"    Open {DB_PATH} in your DB terminal/browser and run the contents of that file.")

if __name__ == "__main__":
    args = sys.argv[1:]
    to_clipboard = False
    
    # 1. Check for clipboard flag anywhere in arguments
    if '-c' in args:
        to_clipboard = True
        args.remove('-c')
        
    # 2. Extract max_new_tokens if the remaining LAST argument is a digit
    max_tokens = 400
    if args and args[-1].isdigit():
        max_tokens = int(args.pop())
        
    # 3. Any remaining args form the prompt
    prompt = " ".join(args) if args else "Explain what an SQL database is."
    
    # 4. Unconditionally build the database every single time
    build_optimized_sql_db()
    
    # 5. Tokenize, inject the prompt, and update hyperparameters
    write_prompt(prompt, max_new_tokens=max_tokens)
        
    # 6. Generate the massive dynamic SQL string and either copy or save it
    generate_and_output_sql(max_tokens, to_clipboard)
