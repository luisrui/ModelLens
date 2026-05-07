"""
Model Family Extraction Utilities

This module provides functions to extract model family names from model name strings.
Model family represents the base architecture/lineage of a model (e.g., Llama, Qwen, Mistral).
"""

import re
from typing import Dict, List, Optional, Tuple
import json
import os

from module.utils.model_profile import get_model_family, load_model_profile

# Manual mapping from raw model names to families
# This handles special cases that pattern matching might miss
MANUAL_MODEL_FAMILY_MAPPING: Dict[str, str] = {
    # BERT variants
    "camembert": "bert",
    "spanbert": "bert",
    "biobert": "bert",
    "scibert": "bert",
    "pubmedbert": "bert",
    "knowbert": "bert",
    "corefbert": "bert",
    "rubert": "bert",
    "exdeberta": "bert",
    "bertem+mtb": "bert",
    "recent+spanbert": "bert",
    "nli_deberta": "bert",
    "nli_roberta": "bert",
    "dg-spanbert-large": "bert",
    "spanbert-large": "bert",
    "ncbi_bert(large) (p)": "bert",
    "scibert (base vocab)": "bert",
    "scibert (scivocab)": "bert",
    "rubert plain": "bert",
    "exdeberta 567m": "bert",
    "knowbert-w+w": "bert",
    
    # Qwen variants
    "bigqwen2.5-52b-instruct": "qwen",
    "bigqwen2": "qwen",
    "100k_fineweb_continued_pretraining_qwen2.5-0.5b-instruct_unsloth_merged_16bit": "qwen",
    "10k_continued_pretraining_qwen2.5-0.5b-instruct_unsloth_merged_16bit": "qwen",
    "40k_continued_pretraining_qwen2.5-0.5b-instruct_unsloth_merged_16bit": "qwen",
    "83k_continued_pretraining_qwen2.5-0.5b-instruct_unsloth_merged_16bit": "qwen",
    "bbai_7b_koenqwendyan": "qwen",
    "bbai_7b_qwen2.5koen": "qwen",
    "bbai_7b_qwendyankoenlo": "qwen",
    "bbai_7b_qwendyancabslaw": "qwen",
    
    # Llama variants
    "videollama": "llama",
    "videollama2": "llama",
    "videollama2.1": "llama",
    "videollama3(7b)": "llama",
    "videollama2.1(7b)": "llama",
    "videollama2 (72b)": "llama",
    "videollama2 72b": "llama",
    "videollama2 7b": "llama",
    "llamagen": "llama",
    "codellama:13b-4bit-quantised": "llama",
    "codellama:7b-4bit-quantised": "llama",
    
    # Other special cases (speech)
    "hubert-b-ls960": "hubert",
    "hubert with libri-light": "hubert",
}

# Known model families and their aliases/patterns
# Order matters: more specific patterns should come before general ones
KNOWN_FAMILIES = [
    # Major LLM families - order matters for disambiguation
    # Put more specific/distinctive families first
    ("deepseek", ["deepseek", "deepseek-r1", "deepseek-v2"]),
    ("llama", ["llama", "llama-2", "llama-3", "llama2", "llama3", "alpaca", "vicuna", "guanaco", "llamagen", "videollama", "videollama2", "videollama3"]),
    ("qwen", ["qwen", "qwen2", "qwen2.5", "qwen1.5", "bigqwen", "qwq", "qwqen"]),
    ("mistral", ["mistral", "mixtral", "mistral-nemo"]),
    ("phi", ["phi", "phi-2", "phi-3", "phi-4", "phi3", "phi4"]),
    ("gemma", ["gemma", "gemma-2", "gemma2", "codegemma"]),
    ("falcon", ["falcon", "falcon-2", "falcon2"]),
    ("yi", ["yi", "yi-1.5", "yi1.5"]),
    ("gpt", ["gpt", "gpt-2", "gpt2", "gpt-j", "gptj", "gpt-neo", "gptneo", "code-davinci", "openai-o1", "openai-o3", "openai-o4", "openai/o3"]),
    ("palm", ["palm", "palm2"]),
    ("gemini", ["gemini"]),
    ("glm", ["glm", "glm4", "chatglm", "chatglm2", "chatglm3"]),
    ("ernie", ["ernie"]),
    ("pythia", ["pythia"]),
    ("xlnet", ["xlnet"]),
    ("bloom", ["bloom", "bloomz"]),
    ("opt", ["opt"]),
    ("mpt", ["mpt"]),
    ("codellama", ["codellama", "code-llama"]),
    ("starcoder", ["starcoder", "starcoder2", "starcoderbase"]),
    ("baichuan", ["baichuan", "baichuan2"]),
    ("internlm", ["internlm", "internlm2", "internvl", "internvl2"]),
    ("solar", ["solar"]),
    ("openchat", ["openchat"]),
    ("zephyr", ["zephyr"]),
    ("hermes", ["hermes", "nous-hermes", "openhermes"]),
    ("dolphin", ["dolphin"]),
    ("orca", ["orca"]),
    ("wizard", ["wizard", "wizardlm", "wizardcoder", "wizardmath"]),
    ("neural", ["neural", "neuralchat"]),
    ("stablelm", ["stablelm", "stable-lm"]),
    ("command", ["command", "command-r"]),
    ("claude", ["claude"]),
    ("cohere", ["cohere"]),
    ("grok", ["grok"]),
    ("minerva", ["minerva"]),
    # All prefixes with >= 3 models (auto-generated)
    ("mn", ["mn"]),
    ("r[2+1]d", ["r[2+1]d"]),
    ("mfann3bv0", ["mfann3bv0"]),
    ("q2", ["q2"]),
    ("tf", ["tf"]),
    ("revbifpn", ["revbifpn"]),
    ("text", ["text"]),
    ("mm1", ["mm1"]),
    ("reflexis", ["reflexis"]),
    ("kstc", ["kstc"]),
    ("ultiima", ["ultiima"]),
    ("deep", ["deep"]),
    ("janus", ["janus"]),
    ("sida", ["sida"]),
    ("sid", ["sid"]),
    ("qanet", ["qanet"]),
    ("two", ["two"]),
    ("fine", ["fine"]),
    ("unireplknet", ["unireplknet"]),
    ("rdnet", ["rdnet"]),
    ("cvt", ["cvt"]),
    ("xmem", ["xmem"]),
    ("3d", ["3d"]),
    ("panoptic", ["panoptic"]),
    ("ministral", ["ministral"]),
    ("acemath", ["acemath"]),
    ("nemotron", ["nemotron"]),
    ("sauerkrautlm", ["sauerkrautlm"]),
    ("sombrero", ["sombrero"]),
    ("r50", ["r50"]),
    ("dit", ["dit"]),
    ("unifiedqa", ["unifiedqa"]),
    ("ul2", ["ul2"]),
    ("san", ["san"]),
    ("st", ["st"]),
    ("test", ["test"]),
    ("uniter", ["uniter"]),
    ("openflamingo", ["openflamingo"]),
    ("motip", ["motip"]),
    ("cmx", ["cmx"]),
    ("uninet", ["uninet"]),
    ("llemma", ["llemma"]),
    ("densenet", ["densenet"]),
    ("pvtv2", ["pvtv2"]),
    ("rednet", ["rednet"]),
    ("mugglemath", ["mugglemath"]),
    ("baezel", ["baezel"]),
    ("bielik", ["bielik"]),
    ("blossom", ["blossom"]),
    ("miniuslight", ["miniuslight"]),
    ("nanolm", ["nanolm"]),
    ("olmo", ["olmo"]),
    ("saka", ["saka"]),
    ("h2o", ["h2o"]),
    ("miscii", ["miscii"]),
    ("openbuddy", ["openbuddy"]),
    ("thea", ["thea"]),
    ("layoutlmv3", ["layoutlmv3"]),
    ("seamlessm4t", ["seamlessm4t"]),
    ("git", ["git"]),
    ("conv", ["conv"]),
    ("vlm", ["vlm"]),
    ("camelidae", ["camelidae"]),
    ("ensemble", ["ensemble"]),
    ("instructblip", ["instructblip"]),
    ("albef", ["albef"]),
    ("plm", ["plm"]),
    ("pllava", ["pllava"]),
    ("idefics", ["idefics"]),
    ("m2d", ["m2d"]),
    ("segnext", ["segnext"]),
    ("nat", ["nat"]),
    ("shift", ["shift"]),
    ("xlm", ["xlm"]),
    ("stm", ["stm"]),
    ("layernas", ["layernas"]),
    ("q2l", ["q2l"]),
    ("pegasus", ["pegasus"]),
    ("bit", ["bit"]),
    ("ar", ["ar"]),
    ("pdo", ["pdo"]),
    ("litv2", ["litv2"]),
    ("pit", ["pit"]),
    ("scrfd", ["scrfd"]),
    ("adaface", ["adaface"]),
    ("slowfast,", ["slowfast,"]),
    ("metamath", ["metamath"]),
    ("imagen", ["imagen"]),
    ("handreader", ["handreader"]),
    ("deaot", ["deaot"]),
    ("ds", ["ds"]),
    ("svtr", ["svtr"]),
    ("singularity", ["singularity"]),
    ("derivative", ["derivative"]),
    ("exaone", ["exaone"]),
    ("fmixia", ["fmixia"]),
    ("gladiator", ["gladiator"]),
    ("mfann3bv1", ["mfann3bv1"]),
    ("magnolia", ["magnolia"]),
    ("pllum", ["pllum"]),
    ("pantheon", ["pantheon"]),
    ("saba1", ["saba1"]),
    ("summer", ["summer"]),
    ("tsunami", ["tsunami"]),
    ("winter", ["winter"]),
    ("aya", ["aya"]),
    ("cursa", ["cursa"]),
    ("dolly", ["dolly"]),
    ("olmner", ["olmner"]),
    ("recurrentgemma", ["recurrentgemma"]),
    ("id", ["id"]),
    ("speechstew", ["speechstew"]),
    ("microsoft", ["microsoft"]),
    ("eat", ["eat"]),
    ("2d", ["2d"]),
    ("sam", ["sam"]),
    ("unet", ["unet"]),
    ("pixart", ["pixart"]),
    ("edm2", ["edm2"]),
    ("ra", ["ra"]),
    ("neo", ["neo"]),
    ("chinchilla", ["chinchilla"]),
    ("branch", ["branch"]),
    ("lxmert", ["lxmert"]),
    ("lyra", ["lyra"]),
    ("vilt", ["vilt"]),
    ("beit", ["beit"]),
    ("tarsier", ["tarsier"]),
    ("vila", ["vila"]),
    ("ppllava", ["ppllava"]),
    ("vinvl", ["vinvl"]),
    ("flava", ["flava"]),
    ("mossformer2", ["mossformer2"]),
    ("rwkv", ["rwkv"]),
    ("llm", ["llm"]),
    ("longformer", ["longformer"]),
    ("twist", ["twist"]),
    ("titannet", ["titannet"]),
    ("aerialformer", ["aerialformer"]),
    ("deeplabv3", ["deeplabv3"]),
    ("cswin", ["cswin"]),
    ("mask", ["mask"]),
    ("debiformer", ["debiformer"]),
    ("muxnet", ["muxnet"]),
    ("erann", ["erann"]),
    ("beats", ["beats"]),
    ("ast", ["ast"]),
    ("crisscross", ["crisscross"]),
    ("mavil", ["mavil"]),
    ("hrformer", ["hrformer"]),
    ("layoutmask", ["layoutmask"]),
    ("dall", ["dall"]),
    ("gdi", ["gdi"]),
    ("mq", ["mq"]),
    ("audioldm", ["audioldm"]),
    ("mgm", ["mgm"]),
    ("aimv2", ["aimv2"]),
    ("seer", ["seer"]),
    ("lit", ["lit"]),
    ("basic", ["basic"]),
    ("sa", ["sa"]),
    ("sequencer2d", ["sequencer2d"]),
    ("ipt", ["ipt"]),
    ("rexnet", ["rexnet"]),
    ("autoformer", ["autoformer"]),
    ("cloformer", ["cloformer"]),
    ("scalenet", ["scalenet"]),
    ("mixnet", ["mixnet"]),
    ("asymmnet", ["asymmnet"]),
    ("vq", ["vq"]),
    ("graphormer", ["graphormer"]),
    ("odise", ["odise"]),
    ("msnet", ["msnet"]),
    ("slowfast", ["slowfast"]),
    ("damomath", ["damomath"]),
    ("gin", ["gin"]),
    ("3ddfa", ["3ddfa"]),
    ("osvos", ["osvos"]),
    ("al", ["al"]),
    ("rstt", ["rstt"]),
    ("nuwa", ["nuwa"]),
    ("laneaf", ["laneaf"]),
    ("dbrx", ["dbrx"]),
    ("deepmind", ["deepmind"]),
    ("rft", ["rft"]),
    ("trocr", ["trocr"]),
    ("musicgen", ["musicgen"]),
    ("mindact", ["mindact"]),
    ("imp", ["imp"]),
    ("hitnet", ["hitnet"]),
    ("aceinstruct", ["aceinstruct"]),
    ("apollo", ["apollo"]),
    ("aura", ["aura"]),
    ("aurora", ["aurora"]),
    ("bellatrix", ["bellatrix"]),
    ("captain", ["captain"]),
    ("chatwaifu", ["chatwaifu"]),
    ("clarus", ["clarus"]),
    ("cleverboi", ["cleverboi"]),
    ("cybernet", ["cybernet"]),
    ("distilled", ["distilled"]),
    ("lexora", ["lexora"]),
    ("ms", ["ms"]),
    ("minthy", ["minthy"]),
    ("nova", ["nova"]),
    ("proto", ["proto"]),
    ("qandoraexp", ["qandoraexp"]),
    ("rewiz", ["rewiz"]),
    ("rusted", ["rusted"]),
    ("smoltulu", ["smoltulu"]),
    ("tinymistral", ["tinymistral"]),
    ("triangulum", ["triangulum"]),
    ("una", ["una"]),
    ("fr", ["fr"]),
    ("inexpertus", ["inexpertus"]),
    ("josie", ["josie"]),
    ("li", ["li"]),
    ("granite", ["granite"]),
    ("smollm", ["smollm", "smollm2"]),
    ("zeus", ["zeus"]),
    ("ece", ["ece"]),
    ("l3", ["l3"]),
    ("kosmos", ["kosmos"]),
    ("llava", ["llava", "llava-video", "llava-onevision"]),
    ("chocolatine", ["chocolatine"]),
    ("prymmal", ["prymmal"]),
    ("magnum", ["magnum"]),
    ("kronos", ["kronos"]),
    ("calme", ["calme"]),
    ("tora", ["tora"]),
    ("flamingo", ["flamingo"]),
    ("blip", ["blip", "blip2", "blip4"]),
    ("flan", ["flan"]),
    ("gal", ["gal"]),
    ("hybrid", ["hybrid"]),
    
    # Text/Embedding models
    ("bert", ["bert", "roberta", "distilbert", "deberta", "albert", "electra", "camembert", "spanbert", "biobert", "scibert", "pubmedbert", "knowbert", "corefbert", "rubert", "exdeberta"]),
    ("t5", ["t5", "flan-t5", "t5-base", "t5-large"]),
    ("e5", ["e5", "multilingual-e5"]),
    ("gte", ["gte"]),
    ("marian", ["marian", "opus-mt", "opus_mt"]),
    
    # Speech/Audio models
    # NOTE: Do not mix wav2vec2/hubert/wavlm into whisper. They are distinct families and
    # wav2vec2-prefixed models should not be classified as whisper.
    ("wav2vec2", ["wav2vec2", "w2v2"]),
    ("wav2vec", ["wav2vec"]),
    ("hubert", ["hubert"]),
    ("wavlm", ["wavlm"]),
    ("whisper", ["whisper"]),
    ("conformer", ["conformer", "squeezeformer", "convformer"]),
    
    # Vision models
    ("deit", ["deit"]),
    ("beit", ["beit"]),
    ("vit", ["vit", "vision-transformer", "vision"]),
    ("clip", ["clip", "openclip"]),
    ("resnet", ["resnet", "resnet50", "resnet101"]),
    ("vgg", ["vgg", "vgg16", "vgg19"]),
    ("swin", ["swin", "swin-transformer", "swinunet"]),
    ("efficientnet", ["efficientnet"]),
    ("mobilenet", ["mobilenet"]),
    ("convnext", ["convnext"]),
    ("segformer", ["segformer"]),
    ("van", ["van"]),
    ("cait", ["cait"]),
    ("xcit", ["xcit"]),
    ("caformer", ["caformer"]),
    ("transnext", ["transnext"]),
    ("redimnet", ["redimnet"]),
    ("light", ["light"]),
    ("point", ["point", "pointnet", "pointnext"]),
    
    # Architecture types (more general)
    ("cnn", ["cnn", "convnet", "conv-net"]),
    ("rnn", ["rnn", "lstm", "gru", "rnnt"]),
    ("gan", ["gan", "generative"]),
    ("diffusion", ["diffusion", "diff", "ddpm", "ddim", "stable-diffusion"]),
    ("mamba", ["mamba", "actionmamba", "sepmamba", "mambamot"]),
    ("moe", ["moe", "mixture-of-experts", "st-moe", "v-moe"]),
    ("transformer", ["transformer", "trans", "xformer"]),
    
    # Other specialized models
    ("code", ["code", "codegen", "codebert", "codegeex", "codet5"]),
    ("video", ["video", "videollama", "videobert", "videoclip", "videollm"]),
    ("retrieval", ["retrieval", "retriever", "rag", "selfrag"]),
    
    # Additional common families found in analysis
    ("moat", ["moat"]),
    ("sjt", ["sjt"]),
    ("wide", ["wide"]),
    ("vicious", ["vicious"]),
    ("mvit", ["mvit", "mvitv2"]),
    ("dans", ["dans"]),
    ("linkbricks", ["linkbricks"]),
    ("resnext", ["resnext"]),
    ("regvit", ["regvit"]),
    ("lamarck", ["lamarck"]),
    ("mischievous", ["mischievous"]),
    ("reasoningcore", ["reasoningcore"]),
    ("bart", ["bart"]),
    ("mplug", ["mplug"]),
    ("hrnet", ["hrnet", "higherhrnet"]),
    ("internimage", ["internimage"]),
    ("efficientvit", ["efficientvit"]),
    ("deit", ["deit"]),
    ("dinat", ["dinat"]),
    ("tinyvit", ["tinyvit"]),
    ("redpajama", ["redpajama"]),
    ("glam", ["glam"]),
    ("gopher", ["gopher"]),
    ("pali", ["pali"]),
    ("ofa", ["ofa"]),
    ("incoder", ["incoder"]),
    ("atlas", ["atlas"]),
    ("resnest", ["resnest"]),
    ("dat", ["dat"]),
    ("alphanet", ["alphanet"]),
    ("regnety", ["regnety"]),
    ("resmlp", ["resmlp"]),
    ("nfnet", ["nfnet"]),
    ("uniformer", ["uniformer"]),
    ("fastervit", ["fastervit"]),
    ("megatron", ["megatron"]),
    ("aspire", ["aspire"]),
    ("calcium", ["calcium"]),
    ("lucie", ["lucie"]),
    ("mita", ["mita"]),
    ("moganet", ["moganet"]),
    ("aot", ["aot"]),
    ("mimicore", ["mimicore"]),
]

# Compile patterns for efficiency
# Use two types of patterns: exact word boundary matches and substring matches
_FAMILY_PATTERNS_EXACT: List[Tuple[str, re.Pattern]] = []
_FAMILY_PATTERNS_SUBSTRING: List[Tuple[str, re.Pattern]] = []

_ALLOW_SHORT_ALIASES = {"yi", "t5", "e5"}
_IGNORE_FAMILIES = {
    # Generic non-architecture tokens that cause frequent false positives
    "distilled",
    "fine",
    "test",
    "text",
    "deep",
}
_KEYWORD_FALLBACK_FAMILIES = {
    # LLM / text
    "deepseek", "llama", "qwen", "mistral", "phi", "gemma", "falcon", "yi", "gpt",
    "glm", "bloom", "opt", "mpt", "starcoder", "baichuan", "internlm", "pythia",
    # Encoders
    "bert", "t5", "e5", "gte", "marian",
    # Speech/audio
    "whisper", "wav2vec2", "wav2vec", "hubert", "wavlm",
    # Vision
    "vit", "deit", "beit", "clip", "resnet", "vgg", "swin", "efficientnet", "mobilenet", "convnext",
}

for family_name, aliases in KNOWN_FAMILIES:
    if family_name in _IGNORE_FAMILIES:
        continue
    # Avoid extremely short auto-generated aliases (e.g., "fr", "st") which tend to
    # match language codes or org prefixes rather than true architecture families.
    filtered_aliases = []
    for a in aliases:
        a2 = str(a)
        if len(a2) < 3 and family_name not in _ALLOW_SHORT_ALIASES:
            continue
        filtered_aliases.append(a2)
    if not filtered_aliases:
        continue

    # Exact matches with word boundaries (more precise)
    pattern_str_exact = r'\b(' + '|'.join(re.escape(a) for a in filtered_aliases) + r')(?:[-_\d.]|$|\b)'
    _FAMILY_PATTERNS_EXACT.append((family_name, re.compile(pattern_str_exact, re.IGNORECASE)))
    
    # Substring matches (for cases like "spanbert" containing "bert")
    # Only for certain families that commonly appear as suffixes/prefixes
    if family_name in ["bert", "llama", "qwen", "gpt", "t5", "vit", "clip", "resnet", "vgg", "transformer", "cnn", "gan", "diffusion", "mamba", "moe", "rnn", "video", "code"]:
        pattern_str_sub = r'(' + '|'.join(re.escape(a) for a in aliases) + r')'
        _FAMILY_PATTERNS_SUBSTRING.append((family_name, re.compile(pattern_str_sub, re.IGNORECASE)))


def extract_family_from_name(model_name: str) -> str:
    """
    Extract the model family from a model name string using intelligent matching.

    Strategy:
    1. Check manual mapping first (handles special cases)
    2. Try exact pattern matching (word boundaries)
    3. Try substring matching (for variants like "spanbert" -> "bert")
    4. Fallback to "unknown" instead of using first token

    Args:
        model_name: The full model name (e.g., "Llama-3.1-8B-Instruct")

    Returns:
        The family name (e.g., "llama") or "unknown" if not recognized
    """
    # Step 0: Use canonical profile as the first source of truth when available.
    try:
        prof_fam = get_model_family(model_name)
        if prof_fam and prof_fam != "unknown":
            return prof_fam
    except Exception:
        pass

    raw = model_name or ""
    name_lower = raw.lower()
    # Prefer matching against repo-name part to avoid classifying org names as families
    # (e.g., "microsoft/wavlm-base" should be "wavlm", not "microsoft").
    repo_part = raw.split("/")[-1] if "/" in raw else raw
    repo_lower = repo_part.lower()
    
    # Step 1: Check manual mapping first (highest priority)
    if model_name in MANUAL_MODEL_FAMILY_MAPPING:
        return MANUAL_MODEL_FAMILY_MAPPING[model_name]
    
    # Step 2: Try exact pattern matching with word boundaries (more precise)
    for family_name, pattern in _FAMILY_PATTERNS_EXACT:
        if repo_lower and pattern.search(repo_lower):
            return family_name
    for family_name, pattern in _FAMILY_PATTERNS_EXACT:
        if pattern.search(name_lower):
            return family_name
    
    # Step 3: Try substring matching for common families that appear as suffixes/prefixes
    # This handles cases like "spanbert" -> "bert", "videollama" -> "llama"
    # But prioritize longer/more specific matches
    matched_families = []
    for family_name, pattern in _FAMILY_PATTERNS_SUBSTRING:
        if repo_lower and pattern.search(repo_lower):
            matched_families.append(family_name)
        elif pattern.search(name_lower):
            matched_families.append(family_name)
    
    # If multiple matches, prefer more specific ones (longer family names)
    if matched_families:
        # Sort by length (longer = more specific) and return the first
        matched_families.sort(key=lambda x: len(x), reverse=True)
        return matched_families[0]
    
    # Step 4: Try to find known family keywords in the model name
    # Check if any known family keyword appears in the name
    # Prioritize longer/more specific matches
    matched_families_keyword = []
    for family_name, aliases in KNOWN_FAMILIES:
        if family_name in _IGNORE_FAMILIES:
            continue
        if family_name not in _KEYWORD_FALLBACK_FAMILIES:
            continue
        for alias in aliases:
            alias_lower = alias.lower()
            if len(alias_lower) < 3 and family_name not in _ALLOW_SHORT_ALIASES:
                continue
            # Use word boundary or check if it's a significant part of the name
            if (alias_lower in repo_lower) or (alias_lower in name_lower):
                # Prefer matches that are at word boundaries or start of name
                if (
                    repo_lower.startswith(alias_lower)
                    or f"-{alias_lower}" in repo_lower
                    or f"_{alias_lower}" in repo_lower
                    or re.search(r"\b" + re.escape(alias_lower) + r"\b", repo_lower)
                    or name_lower.startswith(alias_lower)
                    or f"-{alias_lower}" in name_lower
                    or f"_{alias_lower}" in name_lower
                    or re.search(r"\b" + re.escape(alias_lower) + r"\b", name_lower)
                ):
                    matched_families_keyword.append((family_name, len(alias)))
                    break
    
    if matched_families_keyword:
        # Sort by alias length (longer = more specific) and return the first
        matched_families_keyword.sort(key=lambda x: x[1], reverse=True)
        return matched_families_keyword[0][0]
    
    # Step 5: Fallback to "unknown" instead of using first token
    # This prevents creating too many unique families
    return "unknown"


def build_model2family(model2id: Dict[str, int]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Build model-to-family mapping and family-to-id mapping from model2id.

    Args:
        model2id: Dictionary mapping model names to IDs

    Returns:
        Tuple of (model2family, family2id) dictionaries
    """
    model2family = {}
    family_set = set()

    for model_name in model2id.keys():
        family = extract_family_from_name(model_name)
        model2family[model_name] = family
        family_set.add(family)

    # Create family2id mapping (sorted for reproducibility)
    sorted_families = sorted(family_set)
    family2id = {f: i for i, f in enumerate(sorted_families)}

    return model2family, family2id


def get_family_id(model_name: str, family2id: Dict[str, int], unknown_id: Optional[int] = None) -> int:
    """
    Get the family ID for a given model name.

    Args:
        model_name: The model name
        family2id: Dictionary mapping family names to IDs
        unknown_id: ID to use for unknown families (default: last ID in family2id)

    Returns:
        The family ID
    """
    # Prefer canonical profile family if available.
    try:
        family = get_model_family(model_name)
    except Exception:
        family = "unknown"
    if family == "unknown":
        family = extract_family_from_name(model_name)
    if family in family2id:
        return family2id[family]
    if unknown_id is not None:
        return unknown_id
    # Default to "unknown" family or 0
    return family2id.get("unknown", 0)


def save_family_mappings(data_dir: str, model2id: Dict[str, int]) -> Tuple[str, str]:
    """
    Generate and save family mapping files.

    Args:
        data_dir: Directory to save the mappings
        model2id: Dictionary mapping model names to IDs

    Returns:
        Paths to the saved model2family.json and family2id.json files
    """
    model2family, family2id = build_model2family(model2id)

    model2family_path = os.path.join(data_dir, "model2family.json")
    family2id_path = os.path.join(data_dir, "family2id.json")

    with open(model2family_path, 'w') as f:
        json.dump(model2family, f, indent=2)

    with open(family2id_path, 'w') as f:
        json.dump(family2id, f, indent=2)

    return model2family_path, family2id_path


def load_or_create_family_mappings(data_dir: str, model2id: Dict[str, int]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """
    Load existing family mappings or create them if they don't exist.

    Args:
        data_dir: Directory containing/to contain the mappings
        model2id: Dictionary mapping model names to IDs

    Returns:
        Tuple of (model2family, family2id) dictionaries
    """
    model2family_path = os.path.join(data_dir, "model2family.json")
    family2id_path = os.path.join(data_dir, "family2id.json")

    # If canonical model_profile.json exists, always rebuild from it (then fallback).
    try:
        profile = load_model_profile()
    except Exception:
        profile = {}

    if profile:
        model2family: Dict[str, str] = {}
        family_set = set()
        for name in model2id.keys():
            fam = get_model_family(name, profile)
            if fam == "unknown":
                fam = extract_family_from_name(name)
            model2family[name] = fam
            family_set.add(fam)

        sorted_families = sorted(family_set)
        family2id = {f: i for i, f in enumerate(sorted_families)}

        # Persist for downstream code that expects these files.
        try:
            with open(model2family_path, "w", encoding="utf-8") as f:
                json.dump(model2family, f, indent=2, ensure_ascii=False)
            with open(family2id_path, "w", encoding="utf-8") as f:
                json.dump(family2id, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return model2family, family2id

    # Otherwise, fall back to cached files if present.
    if os.path.exists(model2family_path) and os.path.exists(family2id_path):
        with open(model2family_path, encoding="utf-8") as f:
            model2family = json.load(f)
        with open(family2id_path, encoding="utf-8") as f:
            family2id = json.load(f)
        return model2family, family2id

    # Create new mappings (extractor-only).
    model2family, family2id = build_model2family(model2id)
    save_family_mappings(data_dir, model2id)
    return model2family, family2id


def find_candidates_for_manual_mapping(
    model2id: Dict[str, int],
    min_family_size: int = 1,
    known_keywords: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, str]]]:
    """
    Find model names that might need manual mapping to families.
    
    This function helps identify models that:
    1. Are currently mapped to "unknown" but might belong to a known family
    2. Contain keywords of known families but weren't matched
    3. Belong to small families (might be misclassified)
    
    Args:
        model2id: Dictionary mapping model names to IDs
        min_family_size: Minimum number of models in a family to consider it valid
        known_keywords: List of keywords to check (e.g., ["bert", "qwen", "llama"])
    
    Returns:
        Dictionary mapping potential family names to list of (model_name, current_family) tuples
    """
    if known_keywords is None:
        known_keywords = ["bert", "qwen", "llama", "gpt", "mistral", "phi", "gemma", "yi"]
    
    # Build current family mapping
    model2family = {}
    family_counts = {}
    for model_name in model2id.keys():
        family = extract_family_from_name(model_name)
        model2family[model_name] = family
        family_counts[family] = family_counts.get(family, 0) + 1
    
    candidates = {}
    
    # Find models that contain known keywords but weren't matched
    for keyword in known_keywords:
        keyword_lower = keyword.lower()
        keyword_candidates = []
        
        for model_name in model2id.keys():
            name_lower = model_name.lower()
            current_family = model2family[model_name]
            
            # Check if model contains keyword but wasn't matched to that family
            if keyword_lower in name_lower and current_family != keyword:
                # Skip if already in manual mapping
                if model_name not in MANUAL_MODEL_FAMILY_MAPPING:
                    keyword_candidates.append((model_name, current_family))
        
        if keyword_candidates:
            candidates[keyword] = keyword_candidates[:50]  # Limit to 50 per keyword
    
    # Find models in small families (might be misclassified)
    small_family_models = []
    for model_name, family in model2family.items():
        if family_counts.get(family, 0) <= min_family_size and family != "unknown":
            small_family_models.append((model_name, family))
    
    if small_family_models:
        candidates["small_families"] = small_family_models[:100]  # Limit to 100
    
    return candidates


def save_manual_mapping_template(output_path: str, model2id: Dict[str, int]):
    """
    Generate a template file with candidate model names for manual mapping.
    
    This creates a JSON file that can be edited and then loaded into MANUAL_MODEL_FAMILY_MAPPING.
    
    Args:
        output_path: Path to save the template JSON file
        model2id: Dictionary mapping model names to IDs
    """
    candidates = find_candidates_for_manual_mapping(model2id)
    
    template = {
        "description": "Manual mapping template. Edit this file to add model->family mappings.",
        "mappings": {},
        "candidates_by_keyword": {}
    }
    
    # Add existing manual mappings
    template["mappings"] = MANUAL_MODEL_FAMILY_MAPPING.copy()
    
    # Add candidates organized by keyword
    for keyword, model_list in candidates.items():
        template["candidates_by_keyword"][keyword] = [
            {"model_name": model, "current_family": family}
            for model, family in model_list
        ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    
    print(f"Template saved to {output_path}")
    print(f"Found {sum(len(v) for v in candidates.values())} candidate models for manual mapping")


def load_manual_mapping_from_file(mapping_file: str) -> Dict[str, str]:
    """
    Load manual mappings from a JSON file.
    
    Args:
        mapping_file: Path to JSON file containing manual mappings
        
    Returns:
        Dictionary mapping model names to family names
    """
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and "mappings" in data:
        return data["mappings"]
    elif isinstance(data, dict):
        return data
    else:
        raise ValueError(f"Invalid format in {mapping_file}")


if __name__ == "__main__":
    # Test the family extractor
    test_names = [
        "Llama-3.1-8B-Instruct",
        "DeepSeek-R1-Distill-Qwen-14B",
        "Qwen2.5-72B-Instruct",
        "Mistral-7B-v0.1",
        "phi-4",
        "gemma-2-9b-it",
        "falcon-40b",
        "yi-34b-chat",
        "whisper-large-v3",
        "bert-base-uncased",
        "unknown-model-123",
        "1.5-Pints-16K-v0.1",
        "AceMath-7B-Instruct",
        "spanbert-large",
        "biobert",
        "videollama2",
        "bigqwen2.5-52b-instruct",
    ]

    print("Testing improved family extraction:")
    for name in test_names:
        family = extract_family_from_name(name)
        print(f"  {name:50s} -> {family}")
