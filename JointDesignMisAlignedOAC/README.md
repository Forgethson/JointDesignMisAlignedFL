# Joint Pre-Equalization and Receiver Combining Design for Federated Learning with Misaligned Over-the-Air Computation

This repository is the official implementation of paper Joint Pre-Equalization and Receiver Combining Design for Federated Learning with Misaligned Over-the-Air Computation

## Documentations

* __main.py__: Initialize the simulation system, optimizing the variables, training the learning model
* __joint.py__: Optimize __x__,  __f__, via Algorithm 1
* __Fed.py__: Given learning model, compute the local gradients, update the global model
* __Transmission.py__: Given the local gradients, perform misaligned over-the-air model aggregation in packets

