# Flower image generator

## Setup environment
Use `gcloud auth application-default login` to connect to the VertexAI.

```bash
uv venv
source .venv/bin/activate
```

## Run the sample

The sample generates images of flower in a folder you run it on
in a directory you mention by --directory.
You can specify the number of images in the example.

The command to run:

```bash
genkit start -- uv run --directory py samples/vertex-ai-imagen/src/vertex-ai-imagen.py
```
