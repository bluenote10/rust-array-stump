#!/bin/bash

cargo tarpaulin -v -o Html -o Xml --exclude-files "benchmarks/*"