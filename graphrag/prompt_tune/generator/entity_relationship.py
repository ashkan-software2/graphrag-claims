# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Entity relationship example generation module."""

import asyncio

from graphrag.language_model.protocol.base import ChatModel
from graphrag.prompt_tune.prompt.entity_relationship import (
    ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT,
    ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
    UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT,
)
from graphrag.prompt_tune.template.community_report_summarization import (
    COMMUNITY_REPORT_EXAMPLE_GENERATION_PROMPT
)

MAX_EXAMPLES = 5


async def generate_entity_relationship_examples(
    model: ChatModel,
    persona: str,
    entity_types: str | list[str] | None,
    docs: str | list[str],
    language: str,
    json_mode: bool = False,
) -> list[str]:
    """Generate a list of entity/relationships examples for use in generating an entity configuration.

    Will return entity/relationships examples as either JSON or in tuple_delimiter format depending
    on the json_mode parameter.
    """
    docs_list = [docs] if isinstance(docs, str) else docs
    history = [{"content": persona, "role": "system"}]

    if entity_types:
        entity_types_str = (
            entity_types
            if isinstance(entity_types, str)
            else ", ".join(map(str, entity_types))
        )

        messages = [
            (
                ENTITY_RELATIONSHIPS_GENERATION_JSON_PROMPT
                if json_mode
                else ENTITY_RELATIONSHIPS_GENERATION_PROMPT
            ).format(entity_types=entity_types_str, input_text=doc, language=language)
            for doc in docs_list
        ]
    else:
        messages = [
            UNTYPED_ENTITY_RELATIONSHIPS_GENERATION_PROMPT.format(
                input_text=doc, language=language
            )
            for doc in docs_list
        ]

    messages = messages[:MAX_EXAMPLES]

    tasks = [
        model.achat(message, history=history, json=json_mode) for message in messages
    ]

    responses = await asyncio.gather(*tasks)

    return [str(response.output.content) for response in responses]



async def generate_community_reporter_examples(
    model: ChatModel,
    persona: str,
    entity_relationship_examples : str | list[str] | None,
    community_reporter_role : str | None,
    report_rating_description : str | None,
    docs: str | list[str],
    language: str,
    json_mode: bool = False,
) -> list[str]:
    """Generate a list of examples for use in generating a community report.
    """
    docs_list = [docs] if isinstance(docs, str) else docs
    history = [{"content": persona, "role": "system"}]

    entities_str_examples = []
    relationships_str_examples = []
    for example in entity_relationship_examples:
        entities_str, relationships_str = parse_entity_relationship_examples(example)
        entities_str_examples.append(entities_str)
        relationships_str_examples.append(relationships_str)
    
    if entity_relationship_examples:
        messages = [
            (
                COMMUNITY_REPORT_EXAMPLE_GENERATION_PROMPT
                if json_mode
                else COMMUNITY_REPORT_EXAMPLE_GENERATION_PROMPT
            ).format(persona=persona, role=community_reporter_role, report_rating_description=report_rating_description,
                     entities=entities, relationships=relationships, input_text=doc)
            for doc, entities, relationships in zip(docs_list[:MAX_EXAMPLES], entities_str_examples, relationships_str_examples)
        ]
    else:
        assert False

    tasks = [
        model.achat(message, history=history, json=json_mode) for message in messages
    ]

    responses = await asyncio.gather(*tasks)

    return docs_list[:MAX_EXAMPLES], entities_str_examples, relationships_str_examples, [str(response.output.content) for response in responses]


import random
import re

def parse_entity_relationship_examples(s):
        # Use literal delimiters
    tuple_delim = "{tuple_delimiter}"
    record_delim = "{record_delimiter}"
    completion_delim = "{completion_delimiter}"

    # Step 1: Remove the completion delimiter and split into records
    content = s.split(completion_delim)[0].strip()
    raw_records = [r.strip() for r in content.split(record_delim) if r.strip()]

    entities = []
    relationships = []
    seen_rels = False

    for rec in raw_records:
        # Step 2: Strip outer parentheses
        if rec.startswith("(") and rec.endswith(")"):
            rec = rec[1:-1]

        # Step 3: Split using literal tuple_delimiter
        parts = [p.strip().strip('"\'') for p in rec.split(tuple_delim)]

        if not parts or len(parts) < 4:
            continue  # skip malformed records

        tag = parts[0].lower()

        if tag == "relationship":
            seen_rels = True

        if (tag == "entity" or tag == "entitites") and not seen_rels:
            _, name, _type, desc = parts[:4]
            entities.append((name, desc))
        elif tag == "relationship":
            _, src, tgt, desc = parts[:4]  # just take the first 4 parts
            relationships.append((src, tgt, desc))

    # Step 4: Assign random unique IDs (0–99)
    total = len(entities) + len(relationships)
    ids = random.sample(range(100), total)

    ENTITIES = "\n".join(f"{ids[i]}, {n}, {d}" for i, (n, d) in enumerate(entities))
    RELS = "\n".join(
        f"{ids[len(entities) + i]}, {s}, {t}, {d}" for i, (s, t, d) in enumerate(relationships)
    )

    return ENTITIES, RELS