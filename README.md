SIPmath
================
Colin Smith

### SIPMath Modeler Tools & Metalog Distribution
This library contains sipmath, a Python implementation of the SIPmath 
Modeler Tools for probabilistic modeling and forecasting by 
[Probability Management Inc](https://www.probabilitymanagement.org/) 
and pymetalog, a Python translation of 
[RMetalog](https://github.com/isaacfab/rmetalog).

### SIPmath Modeler Tools

A SIP, or Stochastic Information Packet, is a method of representing
a probability or frequency distribution as an array of samples and
relevant metadata.

SIPmath is the process of performing aritmetic with SIPs directly.
The SIPmath modeler tools allow for intuitive, sophistocated and 
powerful monte-carlo simulations on an enterprise scale. 

SIPs and SIPmath can be used to create risk models that are 

- Actionable: distributions can be easily manipulated arithmetically
- Additive: individual risk models can be easily aggregated into larger,
comprehensive models.
- Auditable: the Open SIPmath Standard allows for the storage of 
unambiguous data with provenance intact.

The purpose of this repository is to bring the capabilities of the 
SIPmath Modeler Tools for Microsoft Excel to Python. Models created
with this library are backwards compatible with the Excel tools, they
can be exported to Excel and vice versa.

More information about SIPs and SIPMath are available WHERE


### The Metalog Distribution

The metalog distribution is a family of continuous, univariate, and 
highly flexible probability distributions, parametrized by CDF data. 
Metalog distributions elegantly address the common need of finding a 
distributional representation of known data, especially when the data
is not well represented by commonly used distributions. 

Metalog Distributions can be used with SIPmath to create powerful
risk models directly from data without known or pre-fit underlying 
distributions. 

The native Python implementation of the Metalog Distributions in this
Library, pymetalog, is a translation of 
[RMetalog](https://github.com/isaacfab/rmetalog) by Isaac Faber.
More information is available in the [paper](http://pubsonline.informs.org/doi/abs/10.1287/deca.2016.0338)
published in Decision Analysis and the
[website](http://www.metalogdistributions.com/).