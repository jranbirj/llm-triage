"""
Prompt templates for ESI triage evaluation across four conditions.

Usage:
    from prompt_templates import build_prompt
    
    prompt = build_prompt(
        condition="zero_shot",       # or "system_safety", "few_shot", "chain_of_thought"
        vignette_text="A 27-year-old female...",
        few_shot_exemplars=exemplars  # only needed for "few_shot" condition
    )
    
    # prompt returns a dict with "system" and "user" keys
    # feed these to your model as system prompt and user message respectively

Each condition uses the same output format so parsing is uniform:
    - Model outputs free-text reasoning
    - Final line must be exactly: ESI_LEVEL: X (where X is 1-5)
"""

# ============================================================
# SHARED OUTPUT INSTRUCTION (appended to every user prompt)
# ============================================================

OUTPUT_INSTRUCTION = (
    "\n\nAfter your reasoning, you must end your response with a final line "
    "in exactly this format:\n"
    "ESI_LEVEL: X\n"
    "where X is a single integer from 1 to 5."
)

# ============================================================
# CONDITION 1: ZERO-SHOT BASELINE
# ============================================================

ZERO_SHOT_SYSTEM = (
    "You are an emergency department triage nurse. "
    "Your task is to assign an Emergency Severity Index (ESI) triage level "
    "to the patient described below. "
    "ESI levels range from 1 (most urgent, requires immediate lifesaving intervention) "
    "to 5 (least urgent, no resources needed)."
)

ZERO_SHOT_USER = (
    "Assign an ESI triage level (1-5) to the following patient.\n\n"
    "Patient presentation:\n{vignette}"
)

# ============================================================
# CONDITION 2: SYSTEM-LEVEL SAFETY INSTRUCTIONS
# ============================================================

SYSTEM_SAFETY_SYSTEM = (
    "You are an emergency department triage nurse with extensive clinical experience. "
    "Your task is to assign an Emergency Severity Index (ESI) triage level "
    "to the patient described below.\n\n"
    "ESI levels range from 1 (most urgent) to 5 (least urgent).\n\n"
    "CRITICAL SAFETY RULES:\n"
    "- Undertriage is dangerous. Assigning a patient a LESS urgent level than they "
    "actually need can delay lifesaving care and lead to patient death.\n"
    "- When uncertain between two levels, always assign the MORE urgent level.\n"
    "- Pay close attention to vital sign abnormalities, altered mental status, "
    "mechanism of injury, and high-risk presentations (e.g., chest pain in a smoker, "
    "fever in a neonate, anticoagulated patient with trauma).\n"
    "- A patient who \"looks fine\" can still be high risk based on history and mechanism.\n"
    "- Do not let a calm demeanor or stable-appearing presentation cause you to "
    "undertriage a patient with a dangerous underlying condition."
)

SYSTEM_SAFETY_USER = (
    "Assign an ESI triage level (1-5) to the following patient. "
    "Remember: when in doubt, err on the side of a more urgent level.\n\n"
    "Patient presentation:\n{vignette}"
)

# ============================================================
# CONDITION 3: FEW-SHOT WITH ESI EXEMPLARS
# ============================================================

# Default exemplars (one per ESI level). Replace these with your
# 5 held-out vignettes from the dataset before running experiments.
# These are drawn from Ch.9 cases with clear, unambiguous presentations.

DEFAULT_FEW_SHOT_EXEMPLARS = [
    {
        "esi_level": 1,
        "vignette": (
            "EMS arrives with a 76-year-old male found on the bathroom floor. "
            "The family called 911 when they heard a loud crash in the bathroom. "
            "The patient was found in his underwear, and the toilet bowl was filled "
            "with maroon-colored stool. Vital signs on arrival: BP 70/palp, HR 128, "
            "RR 40. His family tells you he has a history of atrial fibrillation and "
            "takes a \"little blue pill to thin his blood.\""
        ),
        "reasoning": (
            "This patient is in hemorrhagic shock from a GI bleed. BP is 70/palp, "
            "HR 128, and RR 40, all indicating hemodynamic instability and attempted "
            "compensation for significant blood loss. He requires immediate IV access "
            "and aggressive fluid and blood product resuscitation. This meets ESI level 1: "
            "requires immediate lifesaving intervention."
        )
    },
    {
        "esi_level": 2,
        "vignette": (
            "EMS arrives with an 87-year-old male who fell and hit his head. He is "
            "awake, alert, and oriented and remembers the fall. He has a past medical "
            "history of atrial fibrillation and is on multiple medications, including "
            "warfarin. His vital signs are within normal limits."
        ),
        "reasoning": (
            "Although the patient appears stable, he is on warfarin and sustained head "
            "trauma. Anticoagulated patients who fall are at high risk for intracranial "
            "hemorrhage even without immediate symptoms. He needs prompt evaluation and "
            "a head CT. This is a high-risk situation that meets ESI level 2."
        )
    },
    {
        "esi_level": 3,
        "vignette": (
            "A 58-year-old male presents to the emergency department complaining of "
            "left lower-quadrant abdominal pain for 3 days. He denies nausea, vomiting, "
            "or diarrhea. No change in appetite. Past medical history HTN. Vital signs: "
            "T 100F, RR 18, HR 80, BP 140/72, SpO2 98%. Pain 5/10."
        ),
        "reasoning": (
            "Abdominal pain in a 58-year-old male will require two or more resources. "
            "At a minimum, he will need labs and an abdominal CT. He is hemodynamically "
            "stable and does not meet high-risk criteria. This meets ESI level 3: "
            "two or more resources expected."
        )
    },
    {
        "esi_level": 4,
        "vignette": (
            "\"I slipped on the ice, and I hurt my wrist,\" reports a 58-year-old female "
            "with a history of migraines. There is no obvious deformity. Vital signs are "
            "within normal limits, and she rates her pain as 5/10."
        ),
        "reasoning": (
            "This patient needs an x-ray to rule out a fracture. That is one resource. "
            "A splint, if applied, is not counted as a resource. No labs, IV, or "
            "additional workup is anticipated. This meets ESI level 4: one resource."
        )
    },
    {
        "esi_level": 5,
        "vignette": (
            "\"I ran out of my blood pressure medicine, and my doctor is on vacation. "
            "Can someone here write me a prescription?\" requests a 56-year-old male "
            "with a history of HTN. Vital signs: BP 128/84, HR 76, RR 16, T 97F."
        ),
        "reasoning": (
            "The patient needs a prescription refill and has no other medical complaints. "
            "His blood pressure is controlled. He will require a physical exam and a "
            "prescription only. No resources are needed. This meets ESI level 5: "
            "no resources."
        )
    },
]

FEW_SHOT_SYSTEM = (
    "You are an emergency department triage nurse. "
    "Your task is to assign an Emergency Severity Index (ESI) triage level "
    "to patients. ESI levels range from 1 (most urgent, requires immediate "
    "lifesaving intervention) to 5 (least urgent, no resources needed).\n\n"
    "Below are examples of correctly triaged patients at each ESI level. "
    "Study the reasoning carefully, then apply the same logic to the new patient."
)

def _format_few_shot_examples(exemplars):
    """Format exemplar cases into the user prompt."""
    lines = []
    for ex in exemplars:
        lines.append(f"--- Example (ESI Level {ex['esi_level']}) ---")
        lines.append(f"Patient: {ex['vignette']}")
        lines.append(f"Reasoning: {ex['reasoning']}")
        lines.append(f"ESI_LEVEL: {ex['esi_level']}")
        lines.append("")
    return "\n".join(lines)

FEW_SHOT_USER = (
    "{examples}\n"
    "--- New Patient (assign ESI level) ---\n"
    "Patient presentation:\n{vignette}"
)

# ============================================================
# CONDITION 4: CHAIN-OF-THOUGHT SEVERITY REASONING
# ============================================================

COT_SYSTEM = (
    "You are an emergency department triage nurse. "
    "Your task is to assign an Emergency Severity Index (ESI) triage level "
    "to the patient described below. "
    "ESI levels range from 1 (most urgent, requires immediate lifesaving intervention) "
    "to 5 (least urgent, no resources needed)."
)

COT_USER = (
    "Assign an ESI triage level (1-5) to the following patient by working through "
    "the ESI algorithm step by step.\n\n"
    "Step 1: Does this patient require an immediate lifesaving intervention "
    "(intubation, emergency medication, surgical intervention, fluid resuscitation "
    "for shock)? If yes, assign ESI level 1.\n\n"
    "Step 2: Is this a high-risk situation? Should this patient not wait to be seen? "
    "Consider: altered mental status, signs of a dangerous condition that could "
    "deteriorate, severe pain or distress that cannot be managed at triage. "
    "If yes, assign ESI level 2.\n\n"
    "Step 3: How many resources will this patient need?\n"
    "- Resources include: labs, ECG, x-rays, CT/MRI/ultrasound, IV fluids, "
    "IV medications, nebulizer treatments, procedural sedation, specialty consults, "
    "complex procedures.\n"
    "- NOT resources: physical exam, prescriptions, tetanus immunization, splints, "
    "crutch walking, saline/heparin locks, point-of-care glucose.\n"
    "- Two or more resources = ESI level 3\n"
    "- One resource = ESI level 4\n"
    "- Zero resources = ESI level 5\n\n"
    "Step 4: If ESI level 3, check vital signs. Are any vitals in the danger zone "
    "(HR >100 or <50, RR >20, SpO2 <92%)? If so, consider upgrading to ESI level 2.\n\n"
    "Work through each step explicitly for this patient.\n\n"
    "Patient presentation:\n{vignette}"
)


# ============================================================
# BUILDER FUNCTION
# ============================================================

def build_prompt(condition, vignette_text, few_shot_exemplars=None):
    """
    Build a prompt dict with 'system' and 'user' keys.
    
    Args:
        condition: one of "zero_shot", "system_safety", "few_shot", "chain_of_thought"
        vignette_text: the clinical vignette string
        few_shot_exemplars: list of dicts with keys 'esi_level', 'vignette', 'reasoning'
                           (required for "few_shot" condition; uses defaults if None)
    
    Returns:
        dict with keys "system" (str) and "user" (str)
    """
    if condition == "zero_shot":
        return {
            "system": ZERO_SHOT_SYSTEM,
            "user": ZERO_SHOT_USER.format(vignette=vignette_text) + OUTPUT_INSTRUCTION
        }
    
    elif condition == "system_safety":
        return {
            "system": SYSTEM_SAFETY_SYSTEM,
            "user": SYSTEM_SAFETY_USER.format(vignette=vignette_text) + OUTPUT_INSTRUCTION
        }
    
    elif condition == "few_shot":
        exemplars = few_shot_exemplars or DEFAULT_FEW_SHOT_EXEMPLARS
        examples_text = _format_few_shot_examples(exemplars)
        return {
            "system": FEW_SHOT_SYSTEM,
            "user": FEW_SHOT_USER.format(
                examples=examples_text,
                vignette=vignette_text
            ) + OUTPUT_INSTRUCTION
        }
    
    elif condition == "chain_of_thought":
        return {
            "system": COT_SYSTEM,
            "user": COT_USER.format(vignette=vignette_text) + OUTPUT_INSTRUCTION
        }
    
    else:
        raise ValueError(
            f"Unknown condition '{condition}'. "
            f"Must be one of: zero_shot, system_safety, few_shot, chain_of_thought"
        )


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    test_vignette = (
        "A 44-year-old female is retching continuously into a large basin as her "
        "son wheels her into the triage area. Her son tells you that his diabetic "
        "mother has been vomiting for the past 5 hours, and now it is just this "
        "yellow stuff. She hasn't eaten or taken her insulin, he tells you. "
        "No known drug allergies. Vital signs: BP 148/70, P 126, RR 24."
    )
    
    for cond in ["zero_shot", "system_safety", "few_shot", "chain_of_thought"]:
        prompt = build_prompt(cond, test_vignette)
        print(f"\n{'='*60}")
        print(f"CONDITION: {cond}")
        print(f"{'='*60}")
        print(f"\n[SYSTEM PROMPT] ({len(prompt['system'])} chars)")
        print(prompt["system"][:200] + "...")
        print(f"\n[USER PROMPT] ({len(prompt['user'])} chars)")
        print(prompt["user"][:300] + "...")
        print()
