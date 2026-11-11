import ollama
import pandas as pd
import regex as re
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
import json
import time
import os
from ollama._types import ResponseError

model = 'gemma4'
input_path = 'data/first_qual_sample.csv'
out_file = 'data/fqs_out.csv'

def extract_json(raw: str) -> dict:
    """Models sometimes wrap JSON in markdown fences or add stray text;
    pull out the first {...} block and parse it."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in model output: {raw!r}")
    return json.loads(match.group(0))


SYSTEM_PROMPT = """You are a careful text classifier. Given a short piece of \
social media text, decide whether it is substantively ABOUT race, ethnicity, \
or racial/ethnic identity, discrimination, or relations (not just text that \
happens to mention a nationality or skin color in passing) in the *real human world*.\
Be careful about content that is just about colors (not races), about fictional races \
(like elves), and other uses of the word race (like car races), that is NOT ABOUT RACE.\

You must have clear evidence that the writer is writing about race. Cases where \
the text is very short or simply includes a topic that is often related to race \
are unclear and your output should reflect that uncertainty. \
Only mark a probabilty greeater than .5 if you think it is likely the text \
is about race. Only mark a probablity greater than .8 if a reasonable person \
reading the text would agree that it is about race for sure. Only mark \
a probability greater than .9 if you are very very sure that the writer \
was writing about race. It should be unambiguous in that case.\
 
Respond with ONLY a single JSON object, no other text, no markdown fences, \
in exactly this shape:
 
{"probability": a number between 0 and 1, "excerpt": "an excerpt from the text that demonstrates that it's about race",  "reason": "one short phrase"}
 
"probability" is your estimated probability that the text is about race. \
0 means that you are sure it's not related to race or racism, while 1  \
means you are sure that it is related. \
"excerpt" is a quote from the text that shows it's about race, leave black if the text isn't about race
"reason" a short explanation for why the excerpt shows the text is about race
Be decisive: use values close to 0 or 1 when you are confident, and only use \
values near 0.5 when the text is genuinely ambiguous."""


few_shot_examples = [
    {
        "text": "Campaigns spend over $6.4 billion on ads for 2022 elections, making race one of the most expensive ever",
        "probability": 0.01,
        "excerpt":"elections, making race",
        "reason": "the excerpt shows this concerns spending about an electoral race, not race and racism",
    },
    {
        "text": "Seattle’s Working People Won the Nation’s First Ban on Caste Discrimination",
        "probability": 0.03,
        "excerpt":"Seattle’s Working People",
        "reason": "this is about social class, it doesn't mention race",
    },
    {
        "text": "https://shopperdiet.wordpress.com/2022/10/29/october-29th-anniversary-of-black-thursday/",
        "probability": 0.0,
        "excerpt":"october-29th-anniversary-of-black-thursday/",
        "reason": "just a url that includes the word 'black' in relation to a date--the 1929 stock market crash",
    },
    {
        "text": "2012 Brazilian Grand Prix: Alonso vs Vettel, title on the line. Race finished under the Safety Car after Paul Di Resta crashes. Vettel is World Champion",
        "probability": 0.01,
        "excerpt":"Grand Prix",
        "reason": "the words Grand Prix show it is about a car race not race or racism",
    },
    {
        "text": "REAL NAZIS REVEALED: The rapidly Devolving Democrat-run shithole of NYC is considering RACE in distributing life-saving COVID treatments",
        "probability": 0.99,
        "excerpt":"considering RACE in distributing",
        "reason": "about how people are using race as a factor to make a decision",
    },
    {
        "text": "Well since you're such a dumb and oppressive woman, I'm sure straightening things up would be him telling you how it's gonna really be now!",
        "probability": 0.4,
        "excerpt":"opressive woman",
        "reason": "while 'opressive' is often used in a racial context, it's not clear that this context is about race, it's possible but unlikely",
    },
    {
        "text": "liberals get cancer, its true, its because they themselves are a cancer so its obvious why it happens.",
        "probability": 0.2,
        "excerpt":"liberals get cancer",
        "reason": "this text is about political ideology--'liberals'--not race",
    },    
]


messages = [{"role": "system", "content": SYSTEM_PROMPT}]
for ex in few_shot_examples:
    messages.append({"role": "user", "content": ex["text"]})
    messages.append({
        "role": "assistant",
        "content": json.dumps({
            "probability": ex["probability"],
            "excerpt": ex["excerpt"],
            "reason": ex["reason"],
        }),
    })


def classify_race(target_text, model = 'llama3.2:3b', temperature = 0.0):
    full_message = messages + [{"role": "user", "content": target_text}]
    try:
        response = ollama.chat(model=model, messages=full_message, options={'request_timeout':240, "temperature": temperature})
    except ResponseError as e:
        if temperature > 1.8:
            print("Aborting due to persistent ResponseError")
            probability = 0.5
            reason = f"[parse failed: {repr(e)}]"
            excerpt = f"[parse failed: {repr(e)}]"
            return probability, reason, excerpt
        else:
            print(f"Found response error with temperature {temperature} and text {target_text}")
            return classify_race(target_text, model, temperature = temperature+.1)
    output = response['message']['content']
    try:
        parsed = extract_json(output)
        probability = float(parsed["probability"])
        reason = str(parsed.get("reason", ""))
        excerpt = str(parsed.get("excerpt", ""))
        if excerpt in [x['excerpt'] for x in few_shot_examples]:
            print("Repeated excerpt")
            full_message = full_message + response["message"] + [{"role": "user", "content": "It seems like you've repeated the excerpt from an earlier example, can you try agian?"}]
            response = ollama.chat(model=model, messages=full_message, options={'request_timeout':240, "temperature": temperature})
            parsed = extract_json(output)
            probability = float(parsed["probability"])
            reason = str(parsed.get("reason", ""))
            excerpt = str(parsed.get("excerpt", ""))
        if reason in [x['reason'] for x in few_shot_examples]:
            print("Repeated reason")
            full_message = full_message + response["message"] + [{"role": "user", "content": "It seems like you've repeated the reason from an earlier example, can you try agian?"}]
            response = ollama.chat(model=model, messages=full_message, options={'request_timeout':240, "temperature": temperature})
            parsed = extract_json(output)
            probability = float(parsed["probability"])
            reason = str(parsed.get("reason", ""))
            excerpt = str(parsed.get("excerpt", ""))
    except (ValueError, KeyError, TypeError) as e:
        # fall back gracefully if the model didn't follow the format
        json_reform = [{"role": "user", "content": f'''Can you fix the format of this JSON so taht it can be read \
                        by python exctract_json method. The format is  Return only valid JSON. Do not add markdown or explanations.\
                        {output}'''}]
        response = ollama.chat(model=model, messages=json_reform, options={'request_timeout':240, "temperature": temperature})
        try:
            output = response['message']['content']
            parsed = extract_json(output)
            probability = float(parsed["probability"])
            reason = str(parsed.get("reason", ""))
            excerpt = str(parsed.get("excerpt", ""))
        except (ValueError, KeyError, TypeError) as e:
            probability = 0.5
            reason = f"[parse failed: {repr(e)}]"
            excerpt = f"[parse failed: {repr(e)}]"
    return probability, reason, excerpt


if __name__ == '__main__':
    # to_classify = pd.read_csv('data/full_sub_001_sample2.csv')
    to_classify = pd.read_csv(input_path)
    # numeric_utc = pd.to_numeric(to_classify['created_utc'], errors='coerce')
    # to_classify['created_utc'] = pd.to_datetime(numeric_utc, unit='s', utc=True)

    to_classify = to_classify.dropna(subset=['full_text'])


    columns_to_grab = ['id', 'full_text', 'created_utc', 'permalink', 'url', 'score','about_race_human']
    columns_to_grab = [x for x in columns_to_grab if x in to_classify.columns]

    to_classify = to_classify[columns_to_grab].sample(frac=1).copy()

    # limit to texts longer than 20 characters
    to_classify = to_classify[to_classify.full_text.str.len()>20].reset_index()

    to_classify = to_classify.drop_duplicates(subset='full_text')

    to_classify['probability'] = 0.0
    to_classify['excerpt'] = ''
    to_classify['reason'] = ''

    if os.path.exists(out_file):
        done_rows = pd.read_csv(out_file)
        to_classify = to_classify[~to_classify.full_text.isin(done_rows.full_text)].copy()

    print(f'Processing {to_classify.shape[0]} rows')

    start_time = round_start = time.time()
    i = 0
    for i in range(i, to_classify.shape[0]):
        if i % 100 == 0:
            end_time = time.time()
            print(f'Processed {i} rows total last 100 in {round(end_time - round_start, 2)} seconds')
            print(f'{round(end_time - start_time, 2)} seconds elapsed')
            round_start = time.time()
            to_classify.iloc[i-100:i].to_csv(out_file, header = not os.path.exists(out_file), mode = 'a', index = False)
            last_i = i

        idx = to_classify.index[i]
        target_text = to_classify.at[idx, 'full_text']

        probability, reason, excerpt = classify_race(target_text, model=model)

        to_classify.at[idx, 'probability'] = probability
        to_classify.at[idx, 'excerpt'] = excerpt
        to_classify.at[idx, 'reason'] = reason

    to_classify.iloc[last_i:i].to_csv(out_file, header = not os.path.exists(out_file), mode = 'a', index = False)
    end_time = time.time()
    print(f'Process Complete: Processed {i} rows total last {i % 100} in {round(end_time - round_start, 2)} seconds')
    print(f'{round(end_time - start_time, 2)} seconds elapsed')

    if('about_race_human' in to_classify.columns):
        thresh = 0.79
        predicted_proba = to_classify.dropna(subset=['probability', 'about_race_human']).probability.to_list()
        predicted = [x>thresh for x in predicted_proba]
        actual = to_classify.dropna(subset=['probability', 'about_race_human']).about_race_human.to_list()
        acc = accuracy_score(actual, predicted)
        f1 = f1_score(actual, predicted)
        roc = roc_auc_score(actual, predicted_proba)

        print(f"Ran {model} with {to_classify.shape[0]} texts")
        print(f"Accuracy was {acc} F1 was {f1}, roc was {roc}")

    to_classify.to_csv(f'data/{model}_classify_test.csv', index = False)
