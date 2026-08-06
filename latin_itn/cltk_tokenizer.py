"""
Replica of CLTK v0 Latin Word Tokenizer with Origin Metadata Tracking.
Matches tokenization logic for Latin-BERT while retaining origin metadata 
for lossless Inverse Text Normalization (ITN) detokenization.
"""

import re
import unicodedata
from typing import List, Tuple, NamedTuple, Optional, Any
from latin_itn.config import PUNCT_MAP


# =====================================================================
# PREPROCESSING NORMALIZATION HELPERS (Diacritics & j/v -> i/u)
# =====================================================================

def strip_diacritics(text: str) -> str:
    """Strips macrons, accents, and converts ligatures (æ/œ/Æ/Œ -> ae/oe/Ae/Oe)."""
    text = (
        text.replace("æ", "ae")
        .replace("œ", "oe")
        .replace("Æ", "Ae")
        .replace("Œ", "Oe")
    )
    nfd = unicodedata.normalize("NFD", text)
    filtered = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return unicodedata.normalize("NFC", filtered)


def normalize_iu(text: str) -> str:
    """
    Normalizes Latin text to standard classical i/u orthography while preserving case:
    - uva -> uua, virgo -> uirgo, jam -> iam.
    - Virgo -> Uirgo, Jam -> Iam.
    - Compound -iacere forms: ejicio -> eicio, Ejicio -> Eicio.
    """
    # Handle compound verbs from -iacere (-jic- / -jici- after prefix/vowel -> -ic- / -ici-)
    text = re.sub(r'([a-zA-Z])ji', r'\1i', text)

    # Target only compound -iic- verb forms (eiicio -> eicio, Eiicio -> Eicio)
    text = re.sub(r'([aeiouAEIOU])iic', r'\1ic', text)

    # Convert all remaining j/J to i/I
    text = text.replace('j', 'i').replace('J', 'I')

    # Convert all remaining v/V to u/U
    text = text.replace('v', 'u').replace('V', 'U')

    return text


def normalize_text(text: str, lower: bool = False) -> str:
    """Applies diacritic stripping and j/v to i/u normalization, preserving case by default."""
    if lower:
        text = text.lower()
    return normalize_iu(strip_diacritics(text))


# =====================================================================
# CLTK EXCEPTION LISTS & ENCLITICS
# =====================================================================

QUE_EXCEPTIONS = [
    'quisque', 'quidque', 'quicque', 'quodque', 'cuiusque', 'cuique', 'quemque', 'quamque', 'quoque',
    'quaque', 'quique', 'quaeque', 'quorumque', 'quarumque', 'quibusque', 'quosque', 'quasque',
    'uterque', 'utraque', 'utrumque', 'utriusque', 'utrique', 'utramque', 'utroque',
    'utraeque', 'utrorumque', 'utrarumque', 'utrisque', 'utrosque', 'utrasque',
    'quicumque', 'quidcumque', 'quodcumque', 'cuiuscumque', 'cuicumque', 'quemcumque', 'quamcumque',
    'quocumque', 'quacumque', 'quaecumque', 'quorumcumque', 'quarumcumque', 'quibuscumque',
    'quoscumque', 'quascumque',
    'unusquisque', 'unaquaeque', 'unumquodque', 'unumquidque', 'uniuscuiusque', 'unicuique',
    'unumquemque', 'unamquamque', 'unoquoque', 'unaquaque',
    'plerusque', 'pleraque', 'plerumque', 'plerique', 'pleraeque', 'pleroque', 'pleramque',
    'plerorumque', 'plerarumque', 'plerisque', 'plerosque', 'plerasque',
    'absque', 'abusque', 'adaeque', 'adusque', 'aeque', 'antique', 'atque', 'circumundique', 'conseque',
    'cumque', 'cunque', 'denique', 'deque', 'donique', 'hucusque', 'inique', 'inseque', 'itaque',
    'longinque', 'namque', 'neque', 'oblique', 'peraeque', 'praecoque', 'propinque', 'qualiscumque',
    'quandocumque', 'quandoque', 'quantuluscumque', 'quantumcumque', 'quantuscumque', 'quinque',
    'quomodocumque', 'quomque', 'quotacumque', 'quotcumque', 'quotienscumque',
    'quotiensque', 'quotusquisque', 'quousque', 'relinque', 'simulatque', 'torque', 'ubicumque',
    'ubique', 'undecumque', 'undique', 'usque', 'usquequaque', 'utcumque', 'utercumque', 'utique',
    'utrimque', 'utrobique', 'utrubique'
]

NE_EXCEPTIONS = [
    'absone', 'acharne', 'acrisione', 'acumine', 'adhucine', 'adsuetudine', 'aeetine', 'aeschynomene',
    'aesone', 'agamemnone', 'agmine', 'albane', 'alcyone', 'almone', 'alsine', 'amasene', 'ambitione',
    'amne', 'amoene', 'amymone', 'anadyomene', 'andrachne', 'anemone', 'aniene', 'anne', 'antigone',
    'aparine', 'apolline', 'aquilone', 'arachne', 'arne', 'arundine', 'ascanione', 'asiane', 'asine',
    'aspargine', 'babylone', 'barine', 'bellone', 'belone', 'bene', 'benigne', 'bipenne', 'bizone',
    'bone', 'bubone', 'bulbine', 'cacumine', 'caligine', 'calymne', 'cane', 'carcine', 'cardine',
    'carmine', 'catacecaumene', 'catone', 'cerne', 'certamine', 'chalbane', 'chamaedaphne',
    'chamaemyrsine', 'chaone', 'chione', 'christiane', 'clymene', 'cognomine', 'commagene', 'commune',
    'compone', 'concinne', 'condicione', 'condigne', 'cone', 'confine', 'consone', 'corone', 'crastine',
    'crepidine', 'crimine', 'crine', 'culmine', 'cupidine', 'cyane', 'cydne', 'cyllene', 'cyrene',
    'daphne', 'depone', 'desine', 'dicione', 'digne', 'dine', 'dione', 'discrimine', 'diutine', 'dracone',
    'dulcedine', 'elatine', 'elephantine', 'elleborine', 'epidamne', 'erigone', 'euadne', 'euphrone',
    'euphrosyne', 'examine', 'faune', 'femine', 'feminine', 'ferrugine', 'fine', 'flamine', 'flumine',
    'formidine', 'fragmine', 'fraterne', 'fulmine', 'fune', 'germane', 'germine', 'geryone', 'gorgone',
    'gramine', 'grandine', 'haecine', 'halcyone', 'hammone', 'harundine', 'hedone', 'helene', 'helxine',
    'hermione', 'heroine', 'hesione', 'hicine', 'hicne', 'hierabotane', 'hippocrene', 'hispane',
    'hodierne', 'homine', 'hominesne', 'hortamine', 'hucine', 'humane', 'hunccine', 'huncine', 'iasione',
    'iasone', 'igne', 'imagine', 'immane', 'immune', 'impoene', 'impone', 'importune', 'impune', 'inane',
    'inconcinne', 'indagine', 'indigne', 'inferne', 'inguine', 'inhumane', 'inpone', 'inpune', 'insane',
    'insigne', 'inurbane', 'ismene', 'istucine', 'itone', 'iuuene', 'karthagine', 'labiene',
    'lacedaemone', 'lanugine', 'latine', 'legione', 'lene', 'lenone', 'libidine', 'limine', 'limone',
    'lumine', 'magne', 'maligne', 'mane', 'margine', 'marone', 'masculine', 'matutine', 'medicamine',
    'melpomene', 'memnone', 'mesene', 'messene', 'misene', 'mitylene', 'mnemosyne', 'moderamine', 'moene',
    'mone', 'mortaline', 'mucrone', 'munimine', 'myrmidone', 'ne', 'necne', 'neptune', 'nequene',
    'nerine', 'nocturne', 'nomine', 'nonne', 'nullane', 'numine', 'nuncine', 'nyctimene', 'obscene',
    'obsidione', 'oenone', 'omine', 'omne', 'oppone', 'opportune', 'ordine', 'origine', 'orphne',
    'oxymyrsine', 'paene', 'pallene', 'pane', 'paraetacene', 'patalene', 'pectine', 'pelagine', 'pellene',
    'pene', 'perbene', 'perbenigne', 'peremne', 'perenne', 'perindigne', 'peropportune', 'persephone',
    'phryne', 'pirene', 'pitane', 'plane', 'pleione', 'plene', 'pone', 'praefiscine', 'prasiane',
    'priene', 'priuigne', 'procne', 'proditione', 'progne', 'prone', 'propone', 'pulmone', 'pylene',
    'pyrene', 'pythone', 'ratione', 'regione', 'religione', 'remane', 'retine', 'rhene', 'rhododaphne',
    'robigine', 'romane', 'roxane', 'rubigine', 'sabine', 'sane', 'sanguine', 'saturne', 'seditione',
    'segne', 'selene', 'semine', 'semiplene', 'sene', 'sepone', 'serene', 'sermone', 'serrane', 'siccine',
    'sicine', 'sine', 'sithone', 'solane', 'sollemne', 'somne', 'sophene', 'sperne', 'spiramine',
    'stamine', 'statione', 'stephane', 'sterne', 'stramine', 'subpone', 'subtegmine', 'subtemine',
    'sulmone', 'superne', 'supine', 'suppone', 'susiane', 'syene', 'tantane', 'tantine', 'taprobane',
    'tegmine', 'telamone', 'temne', 'temone', 'tene', 'testudine', 'theophane', 'therone', 'thyone',
    'tiberine', 'tibicine', 'tiburne', 'tirone', 'tisiphone', 'torone', 'transitione', 'troiane',
    'turbine', 'turne', 'tyrrhene', 'uane', 'uelamine', 'uertigine', 'uesane', 'uimine', 'uirgine',
    'umbone', 'unguine', 'uolumine', 'uoragine', 'urbane', 'uulcane', 'zone'
]

N_EXCEPTIONS = [
    'aenean', 'agmen', 'alioquin', 'an', 'attamen', 'cacumen', 'carmen', 'certamen', 'clymenen', 'cognomen',
    'crimen', 'culmen', 'dein', 'deucalion', 'discrimen', 'en', 'epitheton', 'erinyn', 'exin', 'flumen',
    'forsan', 'forsitan', 'fulmen', 'gramen', 'hymen', 'iason', 'in', 'limen', 'liquamen', 'lumen', 'nomen',
    'non', 'numen', 'omen', 'orion', 'paean', 'pan', 'pelion', 'phaethon', 'python', 'quin', 'semen', 'sin',
    'specimen', 'tamen', 'themin', 'titan', 'alcuin', 'caen', 'christian', 'chronicon', 'châtillon', 'claudian',
    'iohn', 'iustin', 'latin', 'lucan', 'martin', 'nouatian', 'quintilian', 'roman', 'tertullian'
]

UE_EXCEPTIONS = [
    'agaue', 'ambigue', 'assidue', 'aue', 'boue', 'breue', 'calue', 'caue', 'ciue', 'congrue', 'contigue',
    'continue', 'curue', 'exigue', 'exue', 'fatue', 'faue', 'fue', 'furtiue', 'gradiue', 'graue',
    'ignaue', 'incongrue', 'ingenue', 'innocue', 'ioue', 'lasciue', 'leue', 'moue', 'mutue', 'naue',
    'neue', 'niue', 'perexigue', 'perspicue', 'pingue', 'praecipue', 'praegraue', 'prospicue', 'proterue',
    'remoue', 'resolue', 'saeue', 'salue', 'siue', 'solue', 'strenue', 'sue', 'summoue', 'superflue',
    'supplicue', 'tenue', 'uiue', 'ungue', 'uoue'
]

ST_EXCEPTIONS = [
    'abest', 'adest', 'ast', 'deest', 'est', 'inest', 'interest', 'post', 'potest', 'prodest', 'subest', 'superest'
]

ENCLITICS = ['que', 'n', 'ne', 'ue', 'st']

LATIN_EXCEPTIONS = set(
    QUE_EXCEPTIONS + NE_EXCEPTIONS + N_EXCEPTIONS + UE_EXCEPTIONS + ST_EXCEPTIONS + ENCLITICS
)

LATIN_REPLACEMENTS_MAP = {
    'mecum': ('cum', 'me'),
    'tecum': ('cum', 'te'),
    'secum': ('cum', 'se'),
    'nobiscum': ('cum', 'nobis'),
    'uobiscum': ('cum', 'uobis'),
    'quocum': ('cum', 'quo'),
    'quacum': ('cum', 'qua'),
    'quicum': ('cum', 'qui'),
    'quibuscum': ('cum', 'quibus'),
    'sodes': ('si', 'audes'),
    'sis': ('si', 'uis'),
    'satin': ('satis', 'ne'),
    'scin': ('scis', 'ne'),
    'uin': ('uis', 'ne'),
    'sultis': ('si', 'uultis'),
    'similist': ('similis', 'est'),
    'qualist': ('qualis', 'est')
}


class TokenInfo(NamedTuple):
    """Container for token string, enclitic flag, and original pre-tokenized/unnormalized word stem."""
    token_str: str
    is_enclitic: bool
    orig_word: Optional[str] = None


class CLTKLegacyLatinTokenizer:
    """
    CLTK v0 word tokenizer with origin metadata tracking for ITN detokenization,
    incorporating Latin preprocessing normalization (diacritic stripping and j/v to i/u).
    Supports case-sensitive tokenization.
    """

    def __init__(self):
        self.enclitics = ENCLITICS
        self.exceptions = LATIN_EXCEPTIONS
        self.replacements_map = LATIN_REPLACEMENTS_MAP

    def tokenize(self, text: str, lower: bool = False) -> List[TokenInfo]:
        """
        Tokenizes string into normalized tokens while tracking origin metadata
        for lossless ITN detokenization (retaining macrons, casing, j/v orthography, etc.).
        Returns: [TokenInfo(token_str, is_enclitic, orig_word), ...]
        """
        raw_tokens = text.strip().split()
        final_tokens: List[TokenInfo] = []

        for token in raw_tokens:
            norm_token = normalize_text(token, lower=lower)
            lowered_norm = norm_token.lower()

            # Check for compound word replacements using lowercase lookup key
            if lowered_norm in self.replacements_map:
                head, tail = self.replacements_map[lowered_norm]
                
                # Match casing of head token to input word capitalization
                if token.isupper():
                    head, tail = head.upper(), tail.upper()
                elif token[0].isupper():
                    head = head.capitalize()

                final_tokens.append(TokenInfo(token_str=head, is_enclitic=False, orig_word=token))
                final_tokens.append(TokenInfo(token_str=tail, is_enclitic=True, orig_word=token))
                continue

            # Check for enclitics if not in exceptions
            is_enclitic = False
            if lowered_norm not in self.exceptions:
                for enclitic in self.enclitics:
                    if lowered_norm.endswith(enclitic):
                        if enclitic == 'n':
                            orig_stem = token[:-1]
                            norm_stem = normalize_text(orig_stem, lower=lower)
                            enc_str = '-NE' if token.isupper() else '-ne'
                            final_tokens.append(TokenInfo(token_str=norm_stem, is_enclitic=False, orig_word=orig_stem))
                            final_tokens.append(TokenInfo(token_str=enc_str, is_enclitic=True, orig_word=token))
                        elif enclitic == 'st':
                            is_ust = lowered_norm.endswith('ust')
                            orig_stem = token[:-1] if is_ust else token[:-2]
                            norm_stem = normalize_text(orig_stem, lower=lower)
                            est_str = 'EST' if token.isupper() else 'est'
                            final_tokens.append(TokenInfo(token_str=norm_stem, is_enclitic=False, orig_word=orig_stem))
                            final_tokens.append(TokenInfo(token_str=est_str, is_enclitic=True, orig_word=token))
                        else:
                            orig_stem = token[:-len(enclitic)]
                            orig_enc = token[-len(enclitic):]
                            norm_stem = normalize_text(orig_stem, lower=lower)
                            norm_enc = f"-{normalize_text(orig_enc, lower=lower)}"

                            orig_word_val = orig_stem if orig_stem != norm_stem else None
                            final_tokens.append(TokenInfo(token_str=norm_stem, is_enclitic=False, orig_word=orig_word_val))
                            final_tokens.append(TokenInfo(token_str=norm_enc, is_enclitic=True, orig_word=None))
                        
                        is_enclitic = True
                        break

            # Fallback for standard words
            if not is_enclitic:
                orig_word_val = token if token != norm_token else None
                final_tokens.append(TokenInfo(token_str=norm_token, is_enclitic=False, orig_word=orig_word_val))

        return final_tokens


def parse_tag(tag: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Parses a predicted tag string (e.g., 'TITLE_PERIOD') into casing directive 
    and punctuation character.
    
    Returns (None, "") if no tag is provided.
    """
    if not tag:
        return None, ""

    parts = tag.split("_", 1)
    casing = parts[0]
    punct_type = parts[1] if len(parts) > 1 else "NONE"
    punct = PUNCT_MAP.get(punct_type, "")

    return casing, punct


def _apply_casing(text: str, casing: Optional[str]) -> str:
    """Applies casing directive to text. Leaves text unchanged if casing is None."""
    if casing == "TITLE":
        return text.capitalize()
    if casing == "LOWER":
        return text.lower()
    return text


def rejoin_enclitics_and_format(
    tokens_with_metadata: List[Tuple[str, bool] | Any],
    predicted_tags: Optional[List[str]] = None,
) -> str:
    """
    Rejoins enclitics and expanded compound forms back to host words
    and optionally applies predicted ITN casing and punctuation.

    If `predicted_tags` is omitted or None, original casing is preserved
    and no punctuation is added.
    """
    formatted_words = []

    # If tags aren't provided, pair each token with None
    tags = predicted_tags if predicted_tags is not None else [None] * len(tokens_with_metadata)
    use_strict = predicted_tags is not None

    for meta, tag in zip(tokens_with_metadata, tags, strict=use_strict):
        # Extract metadata
        token_str = meta[0]
        is_enclitic = meta[1]
        orig_word = getattr(meta, "orig_word", None) or (meta[2] if len(meta) > 2 else None)

        casing, punct = parse_tag(tag)
        clean_token = token_str[1:] if token_str.startswith("-") else token_str

        if is_enclitic and formatted_words:
            prev_word = formatted_words.pop()
            prev_word_clean = prev_word.rstrip(".,?!;: ")

            if orig_word:
                # Determine host word capitalization state
                is_head_capitalized = (
                    prev_word_clean[0].isupper() if prev_word_clean else (casing == "TITLE")
                )

                if casing is not None:
                    base_word = orig_word.capitalize() if is_head_capitalized else orig_word.lower()
                else:
                    # Preserve original orthography casing if no tags provided
                    base_word = orig_word.capitalize() if is_head_capitalized else orig_word

                merged = base_word + punct
            else:
                # Standard attached enclitics ("-que", "-ve")
                word_formatted = _apply_casing(clean_token, casing)
                merged = prev_word_clean + word_formatted + punct

            formatted_words.append(merged)
        else:
            base_target = orig_word if orig_word else clean_token
            word_formatted = _apply_casing(base_target, casing)
            formatted_words.append(word_formatted + punct)

    return " ".join(formatted_words)