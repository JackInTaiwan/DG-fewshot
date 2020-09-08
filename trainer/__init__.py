from . import (
    rel_cross_domain_sampling,
    rel_identical_domain_sampling,
    proto_cross_domain_sampling,
    proto_identical_domain_sampling,
)



TRAINERS = {
    "RelCrossDomainSampling": rel_cross_domain_sampling.RelCrossDomainSamplingTrainer,
    "RelIdenticalDomainSampling": rel_identical_domain_sampling.RelIdenticalDomainSamplingTrainer,
    "ProtoCrossDomainSampling": proto_cross_domain_sampling.ProtoCrossDomainSamplingTrainer,
    "ProtoIdenticalDomainSampling": proto_identical_domain_sampling.ProtoIdenticalDomainSamplingTrainer
}