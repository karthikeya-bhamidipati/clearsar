"""
Inference engine for the ClearSAR Advisor (see knowledge_base.yaml for the rule format).

    kb = KnowledgeBase.load()
    result = InferenceEngine(kb).run(numeric_facts, symbolic_facts, context)

Steps
  1. fuzzify   numeric facts -> symbolic levels with a certainty factor (CF)
  2. chain     rules group by group (stratified forward chaining); a rule fires when all its
               premises hold with CF >= min_premise_cf; conclusion CF = rule.cf * min(premise CFs);
               evidence for the same conclusion combines as cf1 + cf2 * (1 - cf1)  (MYCIN)
  3. resolve   single-valued attributes keep the value with the highest CF
  4. explain   every fired rule is kept in a trace ("how"); rules that did not fire record the
               first premise that failed ("why not")
"""
import os
from dataclasses import dataclass, field

import yaml

KB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge_base.yaml")


class _SafeDict(dict):
    def __missing__(self, key):
        return "?"


def _fmt(template, context):
    try:
        return template.format_map(_SafeDict(context))
    except (ValueError, TypeError):      # e.g. a numeric format applied to "?"
        out = template
        for k, v in context.items():
            out = out.replace("{" + k + "}", str(v))
        return out


@dataclass
class KnowledgeBase:
    meta: dict
    groups: list
    single_valued: set
    thresholds: dict
    labels: dict
    rules: list
    tie_break: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path=KB_PATH):
        with open(path, encoding="utf-8") as f:
            kb = yaml.safe_load(f)
        for r in kb["rules"]:
            if r["group"] not in kb["groups"]:
                raise ValueError(f"Rule {r['id']}: unknown group {r['group']}")
        return cls(kb["meta"], kb["groups"], set(kb.get("single_valued", [])), kb["thresholds"],
                   kb.get("labels", {}), kb["rules"], kb.get("tie_break", {}))

    def best(self, fact, vals):
        """Highest CF wins; exact ties go to the more cautious value (KB tie_break order)."""
        order = self.tie_break.get(fact, [])
        rank = {v: i for i, v in enumerate(order)}
        return max(vals, key=lambda v: (round(vals[v], 9), -rank.get(v, len(order))))

    def fuzzify(self, name, value):
        """Numeric value -> (level, cf, borderline?) using the KB thresholds."""
        levels = self.thresholds[name]
        border = float(self.meta.get("borderline", 0.1))
        bounds = sorted({b for lo, hi in levels.values() for b in (lo, hi)})
        inner = bounds[1:-1]                       # boundaries between two levels
        borderline = any(abs(value - b) < border * b for b in inner)   # within 10% of a threshold
        for level, (lo, hi) in levels.items():
            if lo <= value < hi:
                return str(level), (0.7 if borderline else 1.0), borderline
        return str(list(levels)[-1]), 1.0, False


@dataclass
class FiredRule:
    id: str
    group: str
    conclusion: tuple          # (fact, value)
    cf: float                  # CF contributed by this rule
    premises: list             # [(fact, wanted, actual_value, cf)]
    explanation: str


@dataclass
class InferenceResult:
    facts: dict                                         # fact -> {value: cf}
    resolved: dict                                      # single-valued fact -> (value, cf)
    fired: list = field(default_factory=list)           # [FiredRule]
    not_fired: list = field(default_factory=list)       # [(rule_id, group, conclusion, reason)]
    fuzzified: dict = field(default_factory=dict)       # numeric fact -> (value, level, cf, borderline)

    def value(self, fact):
        if fact in self.resolved:
            return self.resolved[fact]
        vals = self.facts.get(fact)
        if not vals:
            return None, 0.0
        v = max(vals, key=vals.get)
        return v, vals[v]

    def support(self, fact, value):
        """Fired rules that concluded fact=value ("how was this concluded?")."""
        return [r for r in self.fired if r.conclusion == (fact, value)]

    def alternatives(self, fact):
        """All values considered for a fact with their combined CF, best first."""
        vals = self.facts.get(fact, {})
        return sorted(vals.items(), key=lambda kv: -kv[1])

    def warnings(self):
        return {f[len("warning_"):]: max(v.values()) for f, v in self.facts.items()
                if f.startswith("warning_") and v}


class InferenceEngine:
    def __init__(self, kb):
        self.kb = kb
        self.min_cf = float(kb.meta.get("min_premise_cf", 0.2))

    @staticmethod
    def _combine(a, b):
        """MYCIN combination for two confirming certainty factors in [0, 1]."""
        return a + b * (1 - a)

    def run(self, numeric, symbolic=None, context=None):
        facts, fuzz = {}, {}
        for name, value in numeric.items():
            if name in self.kb.thresholds and value is not None:
                level, cf, borderline = self.kb.fuzzify(name, float(value))
                facts[name] = {level: cf}
                fuzz[name] = (float(value), level, cf, borderline)
        for name, value in (symbolic or {}).items():
            if value is not None:
                facts[name] = {str(value): 1.0}
        ctx = dict(context or {})
        result = InferenceResult(facts=facts, resolved={}, fuzzified=fuzz)

        for group in self.kb.groups:
            for rule in [r for r in self.kb.rules if r["group"] == group]:
                self._try_rule(rule, result, ctx)
            for fact in self.kb.single_valued:          # resolve this group's verdicts
                if fact in result.facts and fact not in result.resolved and result.facts[fact]:
                    vals = result.facts[fact]
                    best = self.kb.best(fact, vals)
                    result.resolved[fact] = (best, vals[best])
        return result

    def _premise_cf(self, premise, result):
        fact = premise["fact"]
        if fact in result.resolved:                     # later groups see only the verdict
            v, cf = result.resolved[fact]
            vals = {v: cf}
        else:
            vals = result.facts.get(fact, {})
        wanted = premise["in"] if "in" in premise else [premise["is"]]
        wanted = [str(w) for w in wanted]
        cfs = [vals[w] for w in wanted if w in vals]
        actual = max(vals, key=vals.get) if vals else None
        return (max(cfs) if cfs else 0.0), actual, wanted

    def _try_rule(self, rule, result, ctx):
        premises, cfs = [], []
        for p in rule["if"]:
            cf, actual, wanted = self._premise_cf(p, result)
            premises.append((p["fact"], wanted, actual, cf))
            if cf < self.min_cf:
                reason = (f"{p['fact']} is {actual!r}, needed {' or '.join(wanted)}" if actual is not None
                          else f"{p['fact']} is unknown")
                then = rule["then"]
                result.not_fired.append((rule["id"], rule["group"], (then["fact"], then.get("is", "?")), reason))
                return
            cfs.append(cf)
        then = rule["then"]
        value = then.get("is")
        if "value_from" in then:
            value = ctx.get(then["value_from"])
            if value is None:
                result.not_fired.append((rule["id"], rule["group"], (then["fact"], "?"),
                                         f"{then['value_from']} is not available"))
                return
        value = str(value)
        cf = float(rule["cf"]) * min(cfs)
        bucket = result.facts.setdefault(then["fact"], {})
        bucket[value] = self._combine(bucket.get(value, 0.0), cf)
        result.fired.append(FiredRule(rule["id"], rule["group"], (then["fact"], value), cf, premises,
                                      _fmt(rule.get("because", ""), ctx)))
