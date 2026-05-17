<div align="center">

[<img src="image/logo.png" alt="Logo" width="150" height="150">](https://github.com/tkt-gemini/project-2)

# Project-2

**Psychological Text Classification using NLP**

**[Explore the docs »](https://github.com/tkt-gemini/project-2)**

[View Demo](https://psych-classification-tkt.streamlit.app) ·
[Evaluation Report](https://github.com/tkt-gemini/project-2)
</div>

---

<details open="open">
<summary>Table of Contents</summary>

- [Introduce](#introduce)
- [Dataset](#dataset)
- [Installation](#installation)

</details>

---

## Introduce

### Structure
```text
Project Structure
.
├── archive/
├── app.py
├── config.py
├── helper.py
├── eda.ipynb
├── v1.0.0.py
├── v1.0.1.py
└── README.md
```

## Dataset

**[Reddit Mental health (2018-2019)](https://zenodo.org/records/3941387)**

Low, D. M., Rumker, L., Torous, J., Cecchi, G., Ghosh, S. S., & Talkar, T. (2020). Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study. *Journal of Medical Internet Research*, *22*(10), e22635.

```bibtex
@article{low2020natural,
  title={Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study},
  author={Low, Daniel M and Rumker, Laurie and Torous, John and Cecchi, Guillermo and Ghosh, Satrajit S and Talkar, Tanya},
  journal={Journal of medical Internet research},
  volume={22},
  number={10},
  pages={e22635},
  year={2020},
  publisher={JMIR Publications Inc., Toronto, Canada}
}
```

- **15 mental health subreddits** (r/EDAnonymous, r/addiction, r/alcoholism, r/adhd, r/anxiety, r/autism, r/bipolarreddit, r/bpd, r/depression, r/healthanxiety, r/lonely, r/ptsd, r/schizophrenia, r/socialanxiety, and r/suicidewatch).
- **11 non-mental health subreddits** (r/conspiracy, r/divorce, r/fitness, r/guns, r/jokes, r/legaladvice, r/meditation, r/parenting, r/personalfinance, r/relationships, r/teaching).

## Installation

This project uses a **[Pixi](https://pixi.prefix.dev)**. To set up the runtime environment, follow these steps:

```bash
git clone https://github.com/tkt-gemini/project-2.git
cd project-2

pixi shell
```
