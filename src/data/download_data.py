"""Official FEVER dataset acquisition utilities and reproducible sample generator."""
import json
import logging
import os
import urllib.request
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# Official FEVER dataset distribution URLs (Thorne et al., NAACL 2018)
OFFICIAL_URLS = {
    "train": "https://fever.ai/download/fever/train.jsonl",
    "dev": "https://fever.ai/download/fever/shared_task_dev.jsonl",
    "test": "https://fever.ai/download/fever/shared_task_test.jsonl",
    "wiki_dump": "https://fever.ai/download/fever/wiki-pages.zip"
}


def download_official_dataset(split: str, target_dir: str = "data/raw") -> str:
    """Downloads an official FEVER dataset split if not already present.

    Args:
        split: One of 'train', 'dev', 'test', 'wiki_dump'.
        target_dir: Target directory to save the file.

    Returns:
        Path to downloaded file.
    """
    if split not in OFFICIAL_URLS:
        raise ValueError(f"Unknown split '{split}'. Available: {list(OFFICIAL_URLS.keys())}")

    url = OFFICIAL_URLS[split]
    os.makedirs(target_dir, exist_ok=True)
    filename = Path(url).name
    target_path = Path(target_dir) / filename

    if target_path.exists():
        logger.info(f"File already exists at {target_path}. Skipping download.")
        return str(target_path)

    logger.info(f"Downloading {split} from {url} to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        logger.info(f"Successfully downloaded {target_path}")
        return str(target_path)
    except Exception as e:
        logger.error(f"Failed downloading {url}: {e}")
        raise


def create_sample_fever_dataset(sample_dir: str = "data/sample") -> Dict[str, str]:
    """Generates authentic, verified FEVER-formatted sample datasets for local testing and demonstration.

    Creates:
    - wiki_sample.jsonl: Authentic Wikipedia passages formatted with FEVER lines/sentence IDs.
    - train.jsonl: FEVER claims with SUPPORTS, REFUTES, and NOT ENOUGH INFO labels and gold evidence sets.
    - dev.jsonl: Held-out claims for evaluation and hyperparameter tuning.

    Returns:
        Dictionary of created file paths.
    """
    target_dir = Path(sample_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # 1. Wikipedia sample corpus
    wiki_docs = [
        {
            "id": "Nikolaj_Coster-Waldau",
            "text": "Nikolaj William Coster-Waldau is a Danish actor, producer and screenwriter.",
            "lines": (
                "0\tNikolaj William Coster-Waldau (born 27 July 1970) is a Danish actor, producer and screenwriter.\n"
                "1\tHe graduated from the Danish National School of Performing Arts in Copenhagen in 1993.\n"
                "2\tCoster-Waldau's breakthrough performance in Denmark was the role in the film Nightwatch (1994).\n"
                "3\tHe became widely known for his role as Jaime Lannister in the HBO fantasy series Game of Thrones.\n"
                "4\tIn 2008, he starred as Detective John Amsterdam in the Fox television series New Amsterdam.\n"
                "5\tHe worked with the Fox Broadcasting Company on several productions."
            )
        },
        {
            "id": "Alan_Turing",
            "text": "Alan Mathison Turing was an English mathematician, computer scientist, logician and cryptanalyst.",
            "lines": (
                "0\tAlan Mathison Turing (23 June 1912 – 7 June 1954) was an English mathematician, computer scientist, logician, cryptanalyst, philosopher, and theoretical biologist.\n"
                "1\tTuring was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine.\n"
                "2\tHe is widely considered to be the father of theoretical computer science and artificial intelligence.\n"
                "3\tDuring the Second World War, Turing was a leading participant in war-time codebreaking at Bletchley Park.\n"
                "4\tHe played a pivotal role in cracking intercepted coded messages including the German Enigma machine.\n"
                "5\tTuring was prosecuted in 1952 for homosexual acts; the Labouchere Amendment had made male homosexual activity illegal in the United Kingdom.\n"
                "6\tHe died on 7 June 1954 from cyanide poisoning."
            )
        },
        {
            "id": "Apollo_11",
            "text": "Apollo 11 was the American spaceflight that first landed humans on the Moon.",
            "lines": (
                "0\tApollo 11 was the American spaceflight that first landed humans on the Moon.\n"
                "1\tCommander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969.\n"
                "2\tArmstrong became the first person to step onto the lunar surface six hours and 39 minutes later on July 21.\n"
                "3\tAldrin joined him 19 minutes later, and they spent about two and a quarter hours together exploring the site.\n"
                "4\tCommand module pilot Michael Collins flew the command module Columbia alone in lunar orbit.\n"
                "5\tApollo 11 effectively proved US victory in the Space Race with the Soviet Union."
            )
        },
        {
            "id": "Albert_Einstein",
            "text": "Albert Einstein was a German-born theoretical physicist.",
            "lines": (
                "0\tAlbert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time.\n"
                "1\tBest known for developing the theory of relativity, Einstein also made important contributions to quantum mechanics.\n"
                "2\tHis mass–energy equivalence formula E = mc2 has been called 'the world's most famous equation'.\n"
                "3\tHe received the 1921 Nobel Prize in Physics for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect.\n"
                "4\tEinstein settled in the United States in 1933 and became an American citizen in 1940."
            )
        },
        {
            "id": "Mount_Everest",
            "text": "Mount Everest is Earth's highest mountain above sea level.",
            "lines": (
                "0\tMount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.\n"
                "1\tThe China–Nepal border runs across its summit point.\n"
                "2\tIts elevation of 8,848.86 m was most recently established in 2020 by the Chinese and Nepali authorities.\n"
                "3\tMount Everest attracts many climbers, including highly experienced mountaineers.\n"
                "4\tThe first confirmed ascent was achieved by Sir Edmund Hillary and Tenzing Norgay on 29 May 1953."
            )
        },
        {
            "id": "The_Godfather",
            "text": "The Godfather is a 1972 American crime film directed by Francis Ford Coppola.",
            "lines": (
                "0\tThe Godfather is a 1972 American crime film directed by Francis Ford Coppola, who co-wrote the screenplay with Mario Puzo.\n"
                "1\tIt is based on Puzo's best-selling 1969 novel of the same title.\n"
                "2\tThe film stars Marlon Brando, Al Pacino, James Caan, Richard Castellano, Robert Duvall, Sterling Hayden, John Marley, Richard Conte, and Diane Keaton.\n"
                "3\tIt is the first installment in The Godfather trilogy, chronicling the Corleone family under patriarch Vito Corleone.\n"
                "4\tAt the 45th Academy Awards, the film won Best Picture, Best Actor for Marlon Brando, and Best Adapted Screenplay."
            )
        },
        {
            "id": "Photosynthesis",
            "text": "Photosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy.",
            "lines": (
                "0\tPhotosynthesis is a biological process used by plants and other organisms to convert light energy into chemical energy.\n"
                "1\tThis chemical energy is stored in carbohydrate molecules, such as sugars, which are synthesized from carbon dioxide and water.\n"
                "2\tIn most cases, oxygen is also released as a waste product.\n"
                "3\tMost plants, algae, and cyanobacteria perform photosynthesis; such organisms are called photoautotrophs.\n"
                "4\tPhotosynthesis is largely responsible for producing and maintaining the oxygen content of the Earth's atmosphere."
            )
        },
        {
            "id": "Roman_Empire",
            "text": "The Roman Empire was the post-Republican period of ancient Rome.",
            "lines": (
                "0\tThe Roman Empire was the post-Republican period of ancient Rome.\n"
                "1\tAs a polity, it included large territorial holdings around the Mediterranean Sea in Europe, North Africa, and Western Asia, ruled by emperors.\n"
                "2\tThe first two centuries of the empire saw a period of unprecedented stability and prosperity known as the Pax Romana.\n"
                "3\tRome reached its greatest territorial extent under Emperor Trajan in 117 AD.\n"
                "4\tThe Western Roman Empire collapsed in 476 AD when Romulus Augustulus was deposed by the Germanic warlord Odoacer."
            )
        },
        {
            "id": "Marie_Curie",
            "text": "Marie Curie was a Polish and naturalised-French physicist and chemist who conducted pioneering research on radioactivity.",
            "lines": (
                "0\tMarie Salomea Skłodowska-Curie (7 November 1867 – 4 July 1934) was a Polish and naturalised-French physicist and chemist.\n"
                "1\tShe was the first woman to win a Nobel Prize, the first person to win a Nobel Prize twice, and the only person to win a Nobel Prize in two scientific fields.\n"
                "2\tShe won the 1903 Nobel Prize in Physics with her husband Pierre Curie and Henri Becquerel.\n"
                "3\tIn 1911, she won the Nobel Prize in Chemistry for her discovery of the elements polonium and radium.\n"
                "4\tCurie died in 1934 from aplastic anemia caused by exposure to radiation during her scientific research."
            )
        },
        {
            "id": "DNA",
            "text": "Deoxyribonucleic acid is a polymer composed of two polynucleotide chains that coil around each other to form a double helix.",
            "lines": (
                "0\tDeoxyribonucleic acid (DNA) is a polymer composed of two polynucleotide chains that coil around each other to form a double helix.\n"
                "1\tIt carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses.\n"
                "2\tDNA and ribonucleic acid (RNA) are nucleic acids.\n"
                "3\tThe two DNA strands are also known as polynucleotides since they are composed of simpler monomeric units called nucleotides.\n"
                "4\tThe double-helix model of DNA structure was first published in the journal Nature by James Watson and Francis Crick in 1953."
            )
        }
    ]

    # 2. FEVER Training set
    train_claims = [
        # SUPPORTS
        {
            "id": 1001,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
            "evidence": [[[1, 1, "Nikolaj_Coster-Waldau", 4], [1, 2, "Nikolaj_Coster-Waldau", 5]]]
        },
        {
            "id": 1002,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Alan Turing was an English mathematician who broke German codes at Bletchley Park.",
            "evidence": [[[2, 3, "Alan_Turing", 0], [2, 4, "Alan_Turing", 3]]]
        },
        {
            "id": 1003,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Neil Armstrong was the commander of the Apollo 11 mission.",
            "evidence": [[[3, 5, "Apollo_11", 1]]]
        },
        {
            "id": 1004,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Albert Einstein developed the theory of relativity.",
            "evidence": [[[4, 6, "Albert_Einstein", 1]]]
        },
        {
            "id": 1005,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "The first confirmed ascent of Mount Everest was achieved by Sir Edmund Hillary.",
            "evidence": [[[5, 7, "Mount_Everest", 4]]]
        },
        {
            "id": 1006,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "The Godfather won the Academy Award for Best Picture.",
            "evidence": [[[6, 8, "The_Godfather", 4]]]
        },
        {
            "id": 1007,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Photosynthesis produces oxygen as a waste product.",
            "evidence": [[[7, 9, "Photosynthesis", 2]]]
        },
        {
            "id": 1008,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "The Roman Empire reached its maximum territorial extent under Emperor Trajan.",
            "evidence": [[[8, 10, "Roman_Empire", 3]]]
        },
        {
            "id": 1009,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Marie Curie won Nobel Prizes in two different scientific fields.",
            "evidence": [[[9, 11, "Marie_Curie", 1]]]
        },
        {
            "id": 1010,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "James Watson and Francis Crick published the double-helix structure of DNA in 1953.",
            "evidence": [[[10, 12, "DNA", 4]]]
        },

        # REFUTES
        {
            "id": 2001,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Nikolaj Coster-Waldau graduated from Harvard University in 1993.",
            "evidence": [[[11, 13, "Nikolaj_Coster-Waldau", 1]]]
        },
        {
            "id": 2002,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Alan Turing died of natural causes in 1990.",
            "evidence": [[[12, 14, "Alan_Turing", 6]]]
        },
        {
            "id": 2003,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Buzz Aldrin was the first person to step on the Moon during Apollo 11.",
            "evidence": [[[13, 15, "Apollo_11", 2]]]
        },
        {
            "id": 2004,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Albert Einstein received the Nobel Prize in Physics for developing quantum mechanics.",
            "evidence": [[[14, 16, "Albert_Einstein", 3]]]
        },
        {
            "id": 2005,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Mount Everest is located entirely within the borders of India.",
            "evidence": [[[15, 17, "Mount_Everest", 0], [15, 18, "Mount_Everest", 1]]]
        },
        {
            "id": 2006,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "The Godfather was directed by Steven Spielberg.",
            "evidence": [[[16, 19, "The_Godfather", 0]]]
        },
        {
            "id": 2007,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Photosynthesis is a process that consumes oxygen to produce carbon dioxide.",
            "evidence": [[[17, 20, "Photosynthesis", 1], [17, 21, "Photosynthesis", 2]]]
        },
        {
            "id": 2008,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "The Western Roman Empire fell in the 15th century.",
            "evidence": [[[18, 22, "Roman_Empire", 4]]]
        },
        {
            "id": 2009,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Marie Curie was born in Berlin, Germany.",
            "evidence": [[[19, 23, "Marie_Curie", 0]]]
        },
        {
            "id": 2010,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "DNA is composed of three polynucleotide chains forming a triple helix.",
            "evidence": [[[20, 24, "DNA", 0]]]
        },

        # NOT ENOUGH INFO
        {
            "id": 3001,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Nikolaj Coster-Waldau prefers acting in comedies over fantasy television series.",
            "evidence": [[[21, None, None, None]]]
        },
        {
            "id": 3002,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Alan Turing was an avid chess player throughout his childhood.",
            "evidence": [[[22, None, None, None]]]
        },
        {
            "id": 3003,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Neil Armstrong listened to classical music during the Apollo 11 lunar descent.",
            "evidence": [[[23, None, None, None]]]
        },
        {
            "id": 3004,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Albert Einstein often played the violin when contemplating mathematical problems.",
            "evidence": [[[24, None, None, None]]]
        },
        {
            "id": 3005,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Climbing Mount Everest requires five months of intensive high-altitude training.",
            "evidence": [[[25, None, None, None]]]
        },
        {
            "id": 3006,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "The Godfather is Marlon Brando's favorite film among all his starring roles.",
            "evidence": [[[26, None, None, None]]]
        },
        {
            "id": 3007,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Ancient Romans celebrated Trajan's victory with three weeks of public games.",
            "evidence": [[[27, None, None, None]]]
        },
        {
            "id": 3008,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Marie Curie frequently corresponded with British scientists about radiation risks.",
            "evidence": [[[28, None, None, None]]]
        }
    ]

    # 3. FEVER Development set (Held-out evaluation split)
    dev_claims = [
        {
            "id": 4001,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Jaime Lannister was portrayed by Nikolaj Coster-Waldau in Game of Thrones.",
            "evidence": [[[29, 25, "Nikolaj_Coster-Waldau", 3]]]
        },
        {
            "id": 4002,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "The Enigma machine was cracked by American cryptanalysts in Washington.",
            "evidence": [[[30, 26, "Alan_Turing", 3], [30, 27, "Alan_Turing", 4]]]
        },
        {
            "id": 4003,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Michael Collins flew the command module Columbia alone in lunar orbit.",
            "evidence": [[[31, 28, "Apollo_11", 4]]]
        },
        {
            "id": 4004,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Albert Einstein won his Nobel Prize in Physics in the year 1950.",
            "evidence": [[[32, 29, "Albert_Einstein", 3]]]
        },
        {
            "id": 4005,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "The elevation of Mount Everest was officially measured as 8,848.86 m in 2020.",
            "evidence": [[[33, 30, "Mount_Everest", 2]]]
        },
        {
            "id": 4006,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Francis Ford Coppola refused to watch early test screenings of The Godfather.",
            "evidence": [[[34, None, None, None]]]
        },
        {
            "id": 4007,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Organisms that perform photosynthesis include plants, algae, and cyanobacteria.",
            "evidence": [[[35, 31, "Photosynthesis", 3]]]
        },
        {
            "id": 4008,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Romulus Augustulus founded the Pax Romana in 117 AD.",
            "evidence": [[[36, 32, "Roman_Empire", 2], [36, 33, "Roman_Empire", 4]]]
        },
        {
            "id": 4009,
            "verifiable": "VERIFIABLE",
            "label": "SUPPORTS",
            "claim": "Pierre Curie and Marie Curie won the Nobel Prize in Physics together.",
            "evidence": [[[37, 34, "Marie_Curie", 2]]]
        },
        {
            "id": 4010,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "DNA helices were first observed using ultraviolet microscopy in Cambridge.",
            "evidence": [[[38, None, None, None]]]
        },
        {
            "id": 4011,
            "verifiable": "VERIFIABLE",
            "label": "REFUTES",
            "claim": "Neil Armstrong and Buzz Aldrin explored the Moon for ten consecutive hours.",
            "evidence": [[[39, 35, "Apollo_11", 3]]]
        },
        {
            "id": 4012,
            "verifiable": "NOT VERIFIABLE",
            "label": "NOT ENOUGH INFO",
            "claim": "Alan Turing preferred cycling rather than walking to Bletchley Park.",
            "evidence": [[[40, None, None, None]]]
        }
    ]

    wiki_path = target_dir / "wiki_sample.jsonl"
    train_path = target_dir / "train.jsonl"
    dev_path = target_dir / "dev.jsonl"

    with open(wiki_path, "w", encoding="utf-8") as f:
        for doc in wiki_docs:
            f.write(json.dumps(doc) + "\n")

    with open(train_path, "w", encoding="utf-8") as f:
        for claim in train_claims:
            f.write(json.dumps(claim) + "\n")

    with open(dev_path, "w", encoding="utf-8") as f:
        for claim in dev_claims:
            f.write(json.dumps(claim) + "\n")

    logger.info(f"Sample FEVER datasets generated successfully at {sample_dir}")
    return {
        "wiki_sample": str(wiki_path),
        "train": str(train_path),
        "dev": str(dev_path)
    }
