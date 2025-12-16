# Knowledge-guided pre-trained graph transformer

Neural fingerprints (embeddings) based on a knowledge-guided graph transformer. This model reprsents a novel self-supervised learning framework for the representation learning of molecular graphs, consisting of a novel graph transformer architecture, LiGhT, and a knowledge-guided pre-training strategy.

This model was incorporated on 2024-12-17.Last packaged on 2025-10-22.

## Information
### Identifiers
- **Ersilia Identifier:** `eos8aa5`
- **Slug:** `kgpgt-embedding`

### Domain
- **Task:** `Representation`
- **Subtask:** `Featurization`
- **Biomedical Area:** `Any`
- **Target Organism:** `Any`
- **Tags:** `Descriptor`

### Input
- **Input:** `Compound`
- **Input Dimension:** `1`

### Output
- **Output Dimension:** `2304`
- **Output Consistency:** `Fixed`
- **Interpretation:** Knowledge-driven embedding

Below are the **Output Columns** of the model:
| Name | Type | Direction | Description |
|------|------|-----------|-------------|
| dim_0000 | float |  | Encoding dim index 0 of the embedding |
| dim_0001 | float |  | Encoding dim index 1 of the embedding |
| dim_0002 | float |  | Encoding dim index 2 of the embedding |
| dim_0003 | float |  | Encoding dim index 3 of the embedding |
| dim_0004 | float |  | Encoding dim index 4 of the embedding |
| dim_0005 | float |  | Encoding dim index 5 of the embedding |
| dim_0006 | float |  | Encoding dim index 6 of the embedding |
| dim_0007 | float |  | Encoding dim index 7 of the embedding |
| dim_0008 | float |  | Encoding dim index 8 of the embedding |
| dim_0009 | float |  | Encoding dim index 9 of the embedding |

_10 of 2304 columns are shown_
### Source and Deployment
- **Source:** `Local`
- **Source Type:** `External`
- **DockerHub**: [https://hub.docker.com/r/ersiliaos/eos8aa5](https://hub.docker.com/r/ersiliaos/eos8aa5)
- **Docker Architecture:** `AMD64`, `ARM64`
- **S3 Storage**: [https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos8aa5.zip](https://ersilia-models-zipped.s3.eu-central-1.amazonaws.com/eos8aa5.zip)

### Resource Consumption
- **Model Size (Mb):** `428`
- **Environment Size (Mb):** `5873`
- **Image Size (Mb):** `2112.96`

**Computational Performance (seconds):**
- 10 inputs: `36.12`
- 100 inputs: `71.7`
- 10000 inputs: `-1`

### References
- **Source Code**: [https://github.com/lihan97/KPGT](https://github.com/lihan97/KPGT)
- **Publication**: [https://www.nature.com/articles/s41467-023-43214-1](https://www.nature.com/articles/s41467-023-43214-1)
- **Publication Type:** `Peer reviewed`
- **Publication Year:** `2024`
- **Ersilia Contributor:** [miquelduranfrigola](https://github.com/miquelduranfrigola)

### License
This package is licensed under a [GPL-3.0](https://github.com/ersilia-os/ersilia/blob/master/LICENSE) license. The model contained within this package is licensed under a [Apache-2.0](LICENSE) license.

**Notice**: Ersilia grants access to models _as is_, directly from the original authors, please refer to the original code repository and/or publication if you use the model in your research.


## Use
To use this model locally, you need to have the [Ersilia CLI](https://github.com/ersilia-os/ersilia) installed.
The model can be **fetched** using the following command:
```bash
# fetch model from the Ersilia Model Hub
ersilia fetch eos8aa5
```
Then, you can **serve**, **run** and **close** the model as follows:
```bash
# serve the model
ersilia serve eos8aa5
# generate an example file
ersilia example -n 3 -f my_input.csv
# run the model
ersilia run -i my_input.csv -o my_output.csv
# close the model
ersilia close
```

## About Ersilia
The [Ersilia Open Source Initiative](https://ersilia.io) is a tech non-profit organization fueling sustainable research in the Global South.
Please [cite](https://github.com/ersilia-os/ersilia/blob/master/CITATION.cff) the Ersilia Model Hub if you've found this model to be useful. Always [let us know](https://github.com/ersilia-os/ersilia/issues) if you experience any issues while trying to run it.
If you want to contribute to our mission, consider [donating](https://www.ersilia.io/donate) to Ersilia!
