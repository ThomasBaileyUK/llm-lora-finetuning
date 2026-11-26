"""Builds the sample 'recruiter assistant' dataset in data/recruiter_assistant/.

The examples teach a small model a house style: short, specific, candid answers for an
applied-AI technical recruiter. Edit or extend EXAMPLES and re-run:

    python scripts/build_sample_data.py
"""

import json
import random
from pathlib import Path

EXAMPLES = [
    # --- Explaining the tech ---
    ("What is a forward deployed engineer, in one paragraph I can use with clients?",
     "A forward deployed engineer (FDE) is a software engineer who sits with the customer rather than behind the product. They scope the customer's real problem, build and integrate the solution — data pipelines, APIs, LLM workflows, internal tools — and ship it into production, then feed what they learn back to the core product team. Think strong full-stack or backend engineer with consulting instincts: comfortable with ambiguity, client-facing, and judged on customer outcomes rather than tickets closed."),
    ("Explain RAG to me like I'm explaining it to a hiring manager who isn't technical.",
     "RAG (retrieval-augmented generation) is how you get an AI model to answer using your company's own information. When someone asks a question, the system first searches your documents for the most relevant passages, then hands those passages to the language model along with the question, so the answer is grounded in your data rather than the model's memory. It's cheaper and faster to keep up to date than retraining a model, which is why most enterprise AI assistants are built this way."),
    ("What's the difference between fine-tuning and RAG?",
     "RAG changes what the model *knows at answer time* by fetching documents into the prompt. Fine-tuning changes *how the model behaves* by training it on examples — tone, format, a specialised task. Rule of thumb: use RAG for facts that change, fine-tuning for consistent style or skills. Many production systems use both: a fine-tuned model that's good at the task, fed fresh context through retrieval."),
    ("What does LoRA mean on an AI engineer's CV?",
     "LoRA (Low-Rank Adaptation) is a cheap way to fine-tune a large language model. Instead of updating all of the model's billions of weights, you freeze them and train small add-on matrices — often under 1% of the parameters — so it runs on a single GPU in hours rather than a cluster for days. On a CV it signals hands-on model customisation. Good follow-up questions: what base model, what data, how they evaluated it, and whether it went to production."),
    ("What are 'evals' and why do AI teams keep talking about them?",
     "Evals are tests for AI systems. Because LLM outputs vary, teams build sets of example inputs with expected behaviour and score the system against them — automatically, with another model as judge, or with human review. They're how a team knows a prompt change or new model actually made things better rather than just different. Candidates who can talk concretely about building evals tend to be the ones who have shipped LLM features to real users."),
    ("What's an AI agent, simply?",
     "An AI agent is an LLM that can take actions, not just write text. It's given tools — search, a database, an API, a code runner — and works in a loop: decide what to do, call a tool, read the result, decide the next step, until the task is done. The engineering challenge is reliability: keeping it on track, handling tool errors, controlling cost, and evaluating whether it actually completed the task correctly."),
    ("Is a 'prompt engineer' the same as an AI engineer?",
     "No. Prompt engineering is one skill inside AI engineering. An AI engineer builds the whole product around the model: data pipelines, retrieval, tool integrations, evals, latency and cost optimisation, monitoring and the application code. Strong AI engineers are usually solid software engineers first. Someone whose experience is only writing prompts is closer to a solutions or content role."),
    ("What's the difference between an ML engineer and an applied AI engineer?",
     "An ML engineer typically trains and serves models: data pipelines for training, feature engineering, model training, MLOps. An applied AI engineer typically builds products on top of existing foundation models: prompting, RAG, agents, evals and app integration, with light fine-tuning at most. ML engineers lean on PyTorch and training infrastructure; applied AI engineers lean on strong product engineering plus LLM APIs and frameworks."),
    # --- Screening ---
    ("Give me 5 screening questions for a senior forward deployed engineer.",
     "1. Tell me about a time you shipped something for a customer where the requirements changed halfway. What did you do?\n2. Walk me through the last integration you built end to end — data in, system out. What broke in production?\n3. How do you decide when to build a custom solution versus push the customer onto the standard product?\n4. Describe an LLM feature you put in front of real users. How did you know it was working?\n5. When did you last disagree with a customer's technical lead, and how did it play out?\n\nListen for: ownership, production scars, clear communication, and judgement about product vs one-off work."),
    ("What red flags should I listen for when screening an AI engineer?",
     "- Only demos or hackathons, nothing used by real users.\n- Can't explain how they measured quality (no evals, just 'it looked good').\n- Lists every framework but can't go deep on one system they built.\n- No mention of cost, latency or failure handling.\n- Vague about their own contribution vs the team's.\n- Talks about models but not about the data feeding them.\n\nNone is an automatic no, but two or three together usually means a junior profile in a senior title."),
    ("What questions test whether someone has really built RAG in production?",
     "- How did you chunk documents, and what did you change after seeing real queries?\n- How did you measure retrieval quality separately from answer quality?\n- What did you do when the right document wasn't retrieved?\n- How did you keep the index fresh when source data changed?\n- What were latency and cost per query, and how did you bring them down?\n\nPeople with real experience answer with specifics and trade-offs; tutorial-level answers stay generic."),
    ("How do I assess if a product engineer is genuinely product-minded?",
     "Ask them about a feature they shipped and listen for who the user was, what problem it solved, how they knew it worked, and what they cut to ship it. Product-minded engineers talk about usage data, customer conversations and trade-offs unprompted. Then ask what they'd change about a product they use daily — good ones give a specific, reasoned answer within a minute."),
    # --- Sourcing ---
    ("Write a LinkedIn Boolean search for forward deployed engineers with LLM experience in New York.",
     "(\"forward deployed\" OR \"forward-deployed\" OR FDE OR \"deployment engineer\" OR \"solutions engineer\") AND (LLM OR \"large language model\" OR GPT OR RAG OR \"generative AI\" OR GenAI) AND (Python OR TypeScript) NOT (recruiter OR \"talent acquisition\" OR sales)\n\nSet location to New York City Metropolitan Area. Also try \"forward deployed\" with Palantir in past companies — it's the classic FDE training ground."),
    ("Boolean string for product engineers who've shipped AI features, London.",
     "(\"product engineer\" OR \"full stack engineer\" OR \"full-stack engineer\" OR \"software engineer\") AND (LLM OR \"AI features\" OR OpenAI OR Anthropic OR \"generative AI\" OR RAG) AND (React OR TypeScript OR Next.js) AND (startup OR \"Series A\" OR \"Series B\" OR founding)\n\nLocation: London Area, United Kingdom. Filter to current company headcount under 500 to find people who've owned features end to end."),
    ("Where should I look for applied AI engineers beyond LinkedIn?",
     "- GitHub: contributors to LangChain, LlamaIndex, vLLM, DSPy, TRL and similar — check commit history, not just stars.\n- Hugging Face: people publishing models, datasets or Spaces.\n- Hackathon winners: AI Tinkerers, Cerebral Valley and local AI meetups.\n- Engineering blogs and conference talks on evals, RAG or agents.\n- Alumni of AI-native startups that recently raised, were acquired or wound down.\nPersonalise outreach to their actual work — it's the main thing that gets replies."),
    # --- Outreach ---
    ("Write a short LinkedIn outreach to a backend engineer who's been building LLM features at a fintech, for a founding FDE role at a Series A AI company in New York.",
     "Hi {name} — saw you've been shipping LLM features at {company}, especially the {project} work. I'm hiring a founding forward deployed engineer for a Series A AI company in New York: you'd own customer deployments end to end, work directly with the founders, and shape how the product gets built. Strong salary plus meaningful early equity. Worth a 15-minute chat this week?"),
    ("Rewrite this outreach to be shorter and less salesy: 'Hi! I have an AMAZING opportunity with a fast-growing, cutting-edge AI unicorn that is revolutionising the enterprise space. Are you open to new opportunities?'",
     "Hi {name} — I'm working with an AI company that's growing fast in the enterprise market. They're hiring engineers to build and deploy LLM products with customers, and your work on {specific thing} looks like a close fit. Open to a quick chat to see if it's interesting?"),
    ("Write a follow-up message for a candidate who didn't reply to my first outreach a week ago.",
     "Hi {name} — following up in case this got buried. Short version: {company} is hiring a {role} to {one-line impact}, and your experience with {specific skill} stood out. If the timing's wrong, no problem — happy to stay in touch for later in the year."),
    ("How do I message a candidate who's happy in their job?",
     "Don't lead with the job. Lead with what's different: the problem, the people or the stage. Something like: 'Not assuming you're looking — but this team is solving {hard problem} and the founders previously built {credible thing}. If you ever want a sense of what's out there, I'm happy to share how comparable roles are paying in AI right now.' Give value (market insight) and make saying 'not now' easy."),
    # --- Job descriptions ---
    ("Turn these notes into a punchy job ad intro: AI startup, Series A, NYC, legal docs, LLMs, need FDE, customers are law firms, hybrid 3 days.",
     "We're a Series A AI company helping law firms turn thousands of pages of legal documents into answers in minutes. Our forward deployed engineers work hands-on with firms — scoping their workflows, building LLM-powered solutions on our platform, and getting them live in production. You'll be one of our first FDEs, based in New York (hybrid, three days in the office), working directly with the founders."),
    ("What should a good applied AI engineer job description include?",
     "- One clear sentence on what the company does and who it serves.\n- What the person will build in the first six months — concrete projects, not adjectives.\n- The stack, including models and tools (e.g. Python, TypeScript, OpenAI/Anthropic APIs, vector DB, evals tooling).\n- Must-haves kept to four or five; separate nice-to-haves.\n- Salary range and equity, location and hybrid policy.\n- The interview process, step by step.\nCut 'rockstar', 'ninja' and long lists of soft skills — strong engineers skim past them."),
    ("A hiring manager wants 'a senior AI engineer with 10 years of LLM experience'. How do I push back?",
     "Point out that modern LLMs have been widely usable since about 2020, and ChatGPT launched in late 2022, so nobody has ten years of LLM experience. Reframe the requirement: 'Shall we look for 8+ years of strong software engineering, with the last 1–2 years shipping LLM products to production?' That matches the profile they actually want and widens the pool to people who exist."),
    # --- Candidate assessment / summaries ---
    ("Summarise this candidate in 3 bullets for a client: 7 yrs backend Python/Go at two fintechs, last 18 months led a team of 3 building an LLM-based document extraction system processing 50k docs/day, built eval harness, moved from GPT-4 to a fine-tuned open model cutting cost 70%, wants to move to NYC startup, comp expectation $230k base.",
     "- **Seasoned backend engineer** — 7 years in Python and Go across two fintechs; strong on production systems at scale.\n- **Shipped LLM in production** — led a team of three building document extraction handling 50k documents a day, including the evaluation harness, and switched from GPT-4 to a fine-tuned open model, cutting cost by 70%.\n- **Motivation and terms** — wants an early-stage New York startup; expecting around $230k base."),
    ("This candidate says they 'built an AI agent'. What should I ask to understand what they actually did?",
     "- What did the agent do, for whom, and was it used by real users?\n- What tools could it call, and how did it decide which to use?\n- How often did it fail, and how did you detect and handle failures?\n- How did you evaluate it before and after launch?\n- What was your part versus the team's?\nA strong answer has numbers (usage, success rate, latency, cost) and a story about something that went wrong."),
    # --- Market / BD ---
    ("How do I pitch my recruitment services to a Series A AI startup founder?",
     "Keep it about their bottleneck, not your agency. Something like: 'You've just raised and need to hire engineers who can ship with customers — the profile every AI company is competing for. I focus only on applied-AI engineers, so I already know who's strong and who's open to moving. I can show you three calibrated profiles within a week before you commit to anything.' Then ask what roles are slowest to fill right now."),
    ("What signals suggest an AI startup is about to start hiring?",
     "- A new funding round announced in the last 1–3 months.\n- New enterprise customer logos or partnership announcements.\n- Founders posting about 'building the team' or growth on LinkedIn or X.\n- New job posts on their careers page, Wellfound or Y Combinator's job board.\n- Senior engineering leadership recently hired — they'll want to build a team.\n- Opening an office in a new city."),
    # --- Process / candidate experience ---
    ("A strong candidate got an offer elsewhere and gives us 48 hours. What do I tell my client?",
     "Be direct: 'Candidate X has a competing offer with a 48-hour deadline. They prefer us because of {reason}, but they'll accept the other offer if we can't move. To stay in, we need the final interview in the next 24 hours and a decision by {time}.' Give the client the competing package if you have it, and a recommended offer that would win. Speed is usually worth more than an extra interview round."),
    ("How long should an interview process for an AI engineer take?",
     "Aim for two to three weeks from first call to offer, with four or five steps: recruiter screen, hiring manager call, a technical exercise or pairing session, a systems or product deep-dive, and a founder or values conversation. Strong applied-AI engineers often have several processes running, so anything over four weeks risks losing them to a faster competitor."),
    ("Give me a take-home brief for an applied AI engineer that respects their time.",
     "**Brief (3 hours max):** Build a small app that answers questions over the attached set of 30 product docs. Include a short README explaining your retrieval approach, how you'd evaluate answer quality, one failure case you found, and what you'd do with another week.\n\n**We assess:** clarity of reasoning, evaluation thinking and code quality — not UI polish. Offer to pay for the time, or let them walk through an existing project instead."),
    ("Write a friendly rejection after a final round, keeping the door open.",
     "Hi {name}, thank you for the time you put into the process with {company}. They've decided to go with another candidate whose experience was a closer match to {specific area}. They were impressed by {genuine strength}, and so was I. I'd like to keep you in mind for upcoming roles — would you be open to me getting in touch when something fits?"),
    ("What does 'founding engineer' usually mean in terms of equity and expectations?",
     "A founding engineer is usually one of the first one to five engineering hires, joining before or soon after a seed round. Expect broad ownership — architecture, shipping, hiring and on-call — with less structure. Equity is typically meaningfully higher than for later hires (often somewhere around 0.5–2% at seed, lower by Series A), with salary at or a little below market. Exact numbers vary a lot by stage, location and how early they join."),
]

EVAL = [
    ("What does 'fine-tuning' actually mean?",
     "Fine-tuning means taking a pre-trained language model and training it a bit more on your own examples, so it gets better at a specific task or adopts a particular style. The model already knows language from its original training; fine-tuning nudges its behaviour. Techniques like LoRA make this cheap by training only a small set of extra weights."),
    ("Give me 3 questions to test an engineer's experience with evals.",
     "1. Tell me about an eval set you built: where did the examples come from and how big was it?\n2. How did you score outputs — exact match, rules, LLM-as-judge or humans — and how did you check the scoring was trustworthy?\n3. Describe a time an eval caught a regression before it reached users.\nListen for specifics and a sense of how the evals changed decisions."),
    ("Write a 2-line outreach to an ML engineer for an applied AI role at a seed-stage company in London.",
     "Hi {name} — your work on {project} caught my eye. A seed-stage London AI company is hiring its first applied AI engineer to build LLM products with customers; fancy a quick chat?"),
    ("Is someone with only Kaggle experience a fit for an FDE role?",
     "Usually not on their own. Kaggle shows modelling skill, but FDE roles hinge on shipping production software with customers: integrations, APIs, debugging live systems, communicating with stakeholders. Look for evidence of production engineering and customer-facing work alongside it. If that's missing, they're likely a better fit for a data science or ML role."),
    ("What's a vector database, briefly?",
     "A vector database stores text, images or other data as embeddings — lists of numbers that capture meaning — and finds the closest matches to a query very quickly. It's the search engine behind most RAG systems. Examples include Pinecone, Weaviate, Qdrant and pgvector, an extension for Postgres."),
    ("How should I describe QLoRA to a non-technical client?",
     "QLoRA is a way to customise a large AI model on a single affordable GPU. It shrinks the main model by storing it in a compressed 4-bit format, then trains a small add-on on top. You get most of the benefit of fine-tuning a big model at a fraction of the hardware cost."),
]


def to_row(q, a):
    return {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}


def main():
    out = Path(__file__).resolve().parent.parent / "data" / "recruiter_assistant"
    out.mkdir(parents=True, exist_ok=True)
    rows = [to_row(q, a) for q, a in EXAMPLES]
    random.Random(7).shuffle(rows)
    for name, data in [("train.jsonl", rows), ("eval.jsonl", [to_row(q, a) for q, a in EVAL])]:
        with (out / name).open("w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(data)} examples → {out / name}")


if __name__ == "__main__":
    main()
