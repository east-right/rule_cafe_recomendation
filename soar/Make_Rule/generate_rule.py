# soar/Make_Rule/generate_rule.py

import json
from pathlib import Path

DATA_PATH = Path("../../data/soar_rule_keywords.json")
OUTPUT_DIR = Path("./soar_rules")

BOILERPLATE = """sp {elaborate*top-state*top-state
   (state <s> ^superstate nil)
-->
   (<s> ^top-state <s>)
}

sp {elaborate*state*top-state
   (state <s> ^superstate.top-state <ts>)
-->
   (<s> ^top-state <ts>)
}

sp {elaborate*state*item*down
   (state <s> ^superstate.item <i>)
-->
   (<s> ^item <i>)
}

sp {apply*recommend-cafe*send-to-python
   (state <s> ^operator <o>
              ^io.output-link <ol>)
   (<o> ^name recommend-cafe
        ^cafe-name <c-name>)
-->
   (<ol> ^final-recommendation <c-name>)
}
"""


def make_operator(title: str, operator_keywords: list[str]) -> str:
    kw_list = " ".join([f"|{kw}|" for kw in operator_keywords])
    return f"""sp {{recommend*OPERATOR*{title}
   (state <s> ^io.input-link <il>)
   (<il> ^cafe <c>)
   (<c> ^name <c-name> ^keyword << {kw_list} >>)
-->
   (<s> ^operator <o> +)
   (<o> ^name recommend-cafe ^cafe-name <c-name>)
}}
"""


def make_s1_judge(title: str, keyword: str) -> str:
    return f"""sp {{recommend*S1*judge*{title}
   (state <s> ^operator <o1> +
              ^operator <o2> +
              ^io.input-link <il>)
   (<o1> ^name recommend-cafe ^cafe-name <c1-name>)
   (<o2> ^name recommend-cafe ^cafe-name <c2-name> <> <c1-name>)
   (<il> ^cafe <c1> ^cafe <c2>)
   (<c1> ^name <c1-name> ^keyword |{keyword}|)
   (<c2> ^name <c2-name>)
   - {{ (<c2> ^keyword |{keyword}|) }}
-->
   (<s> ^operator <o1> > <o2>)
}}
"""


def make_sn_judge(title: str, keyword: str, depth: int) -> str:
    superstate_chain = ""
    for i in range(1, depth + 1):
        if i == 1:
            superstate_chain += f"   (<s{i}> ^superstate nil)\n"
        else:
            superstate_chain += f"   (<s{i}> ^superstate <s{i-1}>)\n"

    return f"""sp {{resolve*tie*S{depth}*judge*{title}
   (state <s> ^impasse <any-impasse>
              ^superstate <s{depth}>
              ^item <o1> ^item <o2>
              ^top-state <ts>)
{superstate_chain}
   (<o1> ^name recommend-cafe ^cafe-name <c1-name>)
   (<o2> ^name recommend-cafe ^cafe-name <c2-name> <> <c1-name>)
   (<ts> ^io.input-link <il>)
   (<il> ^cafe <c1> ^cafe <c2>)
   (<c1> ^name <c1-name> ^keyword |{keyword}|)
   (<c2> ^name <c2-name>)
   - {{ (<c2> ^keyword |{keyword}|) }}
-->
   (<s{depth-1}> ^operator <o1> > <o2>)
}}
"""


def make_llm_fallback(title: str, depth: int) -> str:
    superstate_chain = ""
    for i in range(1, depth + 1):
        if i == 1:
            superstate_chain += f"   (<s{i}> ^superstate nil)\n"
        else:
            superstate_chain += f"   (<s{i}> ^superstate <s{i-1}>)\n"

    return f"""sp {{resolve*tie*S{depth}*ask-llm*{title}
   (state <s> ^impasse <any-impasse>
              ^superstate <s{depth}>
              ^item <o1> ^item <o2>
              ^top-state <ts>)
{superstate_chain}
   (<o1> ^cafe-name <c1-name>)
   (<o2> ^cafe-name {{ <c2-name> <> <c1-name> }})
   (<ts> ^io.output-link <ol>)
-->
   (<ol> ^ask-llm <req>)
   (<req> ^cand1 <c1-name>
          ^cand2 <c2-name>
          ^stopped-depth {depth})
}}
"""


def generate_soar_rule(rule: dict) -> str:
    title = rule["title"].replace(" ", "_")
    operator_keywords = rule["operator_keywords"]
    tiebreak_keywords = rule["tiebreak_keywords"]

    parts = [BOILERPLATE]
    parts.append(make_operator(title, operator_keywords))

    if tiebreak_keywords:
        parts.append(make_s1_judge(title, tiebreak_keywords[0]))
        for i, kw in enumerate(tiebreak_keywords[1:], start=2):
            parts.append(make_sn_judge(title, kw, i))
        llm_depth = len(tiebreak_keywords) + 1
    else:
        llm_depth = 1

    parts.append(make_llm_fallback(title, llm_depth))

    return "\n".join(parts)


def main():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        rules = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for rule in rules:
        title = rule["title"].replace(" ", "_")
        soar_content = generate_soar_rule(rule)
        output_path = OUTPUT_DIR / f"{title}.soar"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(soar_content)

    print(f"완료: {len(rules)}개 .soar 파일 생성 → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()