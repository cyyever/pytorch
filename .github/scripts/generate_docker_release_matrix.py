#!/usr/bin/env python3

"""Generates a matrix for docker releases through github actions

Will output a condensed version of the matrix. Will include fllowing:
    * CUDA version short
    * CUDA full version
    * CUDNN version short
    * Image type either runtime or devel
    * Platform linux/amd64

"""

import json

import generate_binary_build_matrix


DOCKER_IMAGE_TYPES = ["runtime", "devel"]


def generate_docker_matrix() -> dict[str, list[dict[str, str]]]:
    ret: list[dict[str, str]] = []
    # CUDA amd64 Docker images are available as both runtime and devel.
    for cuda, version in generate_binary_build_matrix.CUDA_ARCHES_FULL_VERSION.items():
        for image in DOCKER_IMAGE_TYPES:
            if (
                image == "devel"
                and cuda in generate_binary_build_matrix.CUDA_ARCHES_RUNTIME_IMAGE_ONLY
            ):
                continue
            ret.append(
                {
                    "cuda": cuda,
                    "cuda_full_version": version,
                    "cudnn_version": generate_binary_build_matrix.CUDA_ARCHES_CUDNN_VERSION[
                        cuda
                    ],
                    "image_type": image,
                    "platform": "linux/amd64",
                }
            )
    return {"include": ret}


if __name__ == "__main__":
    build_matrix = generate_docker_matrix()
    print(json.dumps(build_matrix))
