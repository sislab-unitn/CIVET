# CIVET: Systematic Evaluation of Understanding in VLMs
Repository for the code of the paper *[CIVET: Systematic Evaluation of Understanding in VLMs](https://aclanthology.org/2025.findings-emnlp.239/)* presented at EMNLP 2025.

# Usage

## Generate Stimuli
```shell
cd world
python generate_<experiment>.py DATA_FOLDER
```

## Generate Model Responses

```shell
python -m main MODEL_NAME DATA_FOLDER EXPERIMENT_NAME generate --instruction "Answer using as few words as possible"
```

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This work is licensed under a [MIT License](https://opensource.org/licenses/MIT).

# How To Cite
```
@inproceedings{rizzoli-etal-2025-civet,
    title = "{CIVET}: Systematic Evaluation of Understanding in {VLM}s",
    author = "Rizzoli, Massimo  and
      Alghisi, Simone  and
      Khomyn, Olha  and
      Roccabruna, Gabriel  and
      Mousavi, Seyed Mahed  and
      Riccardi, Giuseppe",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.239/",
    pages = "4462--4480",
    ISBN = "979-8-89176-335-7",
    abstract = "While Vision-Language Models (VLMs) have achieved competitive performance in various tasks, their comprehension of the underlying structure and semantics of a scene remains understudied. To investigate the understanding of VLMs, we study their capability regarding object properties and relations in a controlled and interpretable manner. To this scope, we introduce CIVET, a novel and extensible framework for systematiC evaluatIon Via controllEd sTimuli. CIVET addresses the lack of standardized systematic evaluation for assessing VLMs' understanding, enabling researchers to test hypotheses with statistical rigor. With CIVET, we evaluate five state-of-the-art VLMs on exhaustive sets of stimuli, free from annotation noise, dataset-specific biases, and uncontrolled scene complexity. Our findings reveal that 1) current VLMs can accurately recognize only a limited set of basic object properties; 2) their performance heavily depends on the position of the object in the scene; 3) they struggle to understand basic relations among objects. Furthermore, a comparative evaluation with human annotators reveals that VLMs still fall short of achieving human-level accuracy."
}
```
