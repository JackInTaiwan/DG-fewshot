from . import (
    rel_cross_domain_sampling,
    rel_identical_domain_sampling,
    mix_cross_domain_sampling,
    identical_cross_domain_sampling,
    proto_cross_domain_sampling,
    proto_identical_domain_sampling,
    proto_mix_domain_sampling,
    proto_pseudo_domain_sampling,
    proto_drop_cross_domain_sampling,
)



TRAINERS = {
    "RelCrossDomainSampling": rel_cross_domain_sampling.RelCrossDomainSamplingTrainer,
    "RelIdenticalDomainSampling": rel_identical_domain_sampling.RelIdenticalDomainSamplingTrainer,
    "MixCrossDomainSampling": mix_cross_domain_sampling.MixCrossDomainSamplingTrainer,
    "IdenticalCrossDomainSampling": identical_cross_domain_sampling.IdenticalCrossDomainSamplingTrainer,
    "ProtoCrossDomainSampling": proto_cross_domain_sampling.ProtoCrossDomainSamplingTrainer,
    "ProtoIdenticalDomainSampling": proto_identical_domain_sampling.ProtoIdenticalDomainSamplingTrainer,
    "ProtoMixDomainSampling": proto_mix_domain_sampling.ProtoMixDomainSamplingTrainer,
    "ProtoPseudoDomainSampling": proto_pseudo_domain_sampling.ProtoPseudoDomainSamplingTrainer,
    "ProtoCrossDomainBDBSampling": proto_drop_cross_domain_sampling.ProtoCrossDomainSamplingTrainer,
}