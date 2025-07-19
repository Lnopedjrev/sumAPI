# Overview

API for the summarization model built with FastAPI, Pytriton and Cassandra for use by other applications
Summarization model is simply Mamba2: [link](https://huggingface.co/AntonV/mamba2-1.3b-hf)

The API itself is used by other services, HTTP endpoints are exposed with FastAPI. An article/material to summarize is transferred to Pytriton server via gRPC and written in Cassandra cluster for future analysis and fine-tuning

This package includes components licensed under BSD‑3‑Clause. See licenses/BSD-3-Clause.txt
