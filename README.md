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


### The Metalog Distribution - is this accurate?

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

### Using the package -- WIP

Installation with PyPl (pip) forthcoming

A SIPmath model, sipmodel, is the container class for individual SIPs. 
All SIPs are a part of a larger model. Models only take the number of 
trials as input - this is the number of independent samples from child 
SIP inputs.
```python
import sipmath as sm

sipmodel = sm.sipmodel(trials=10000)
```

Individual SIPs, sipinputs, are the composite distributions of the
sipmodel. There can be as many as necessary for the simulation.
Sipinputs are called from a parent model and take as arguments
distribution parameters and SIP metadata discussed later. 

Currently the supported distributions are:

'uniform','normal','beta','binomial','chisquared','exponential','f',
'discrete','gamma','lognormal','poisson','triangular','t','weibull_min',
'correlated_normal','correlated_uniform','metalog'

All distributions except metalogs, discussed later, are generated using
the package scipy. Please refer to 
[scipy docs](https://docs.scipy.org/doc/scipy/reference/stats.html) 
for parameters used in each distribution. 

Samples are generated by passing a random uniform number [0,1] through
an inverse survival function of the underlying distribution. The
default random number generator is the standard numpy generator, (CHECK)
Hubbard Decision Research's HDR Generator LINK is also supported and
can be set with the generator parameter.

Sipinputs are instantiated as 0 arrays, the model must be manually
be sampled to update all sipinputs simultaneously. 

```python
input_1 = sipmodel.sipinput(distribution='normal', loc=0, scale=1, name='input_1', generator='hdr')

sipmodel.sample()

plt.hist(input_1, 100)
##TODO SHOW PLOT
```

Except in special cases, a sipinput is a one dimensional array
consisting of n=simpmodel.trials independent samples from a 
distribution.

Sipinputs inherit from numpy.ndarray, which allows for simply 
elementwise arithmetic. 

```python
input_1 = sipmodel.sipinput(distribution='uniform')
input_2 = sipmodel.sipinput(distribution='uniform')

sipmodel.sample()

plt.hist(input_1+input_2, 100)
##TODO SHOW PLOT
```




Once the package is loaded you start with a data set of continuous
observations. For this repository, we will load the library and use an
example of fish size measurements from the Pacific Northwest. This data
set is illustrative to demonstrate the flexibility of the metalog
distribution as it is bi-modal. The data is installed with the package.

``` r
library(rmetalog)
data("fishSize")
summary(fishSize)
#>     FishSize   
#>  Min.   : 3.0  
#>  1st Qu.: 7.0  
#>  Median :10.0  
#>  Mean   :10.2  
#>  3rd Qu.:12.0  
#>  Max.   :33.0
```

The base function for the package to create distributions is:

``` r
metalog()
```

This function takes several inputs:

  - x - vector of numeric data
  - term\_limit - integer between 3 and 30, specifying the number of
    metalog distributions, with respective terms, terms to build
    (default: 13)
  - bounds - numeric vector specifying lower or upper bounds, none
    required if the distribution is unbounded
  - boundedness - character string specifying unbounded, semi-bounded
    upper, semi-bounded lower or bounded; accepts values u, su, sl and b
    (default: ‘u’)
  - term\_lower\_bound - (Optional) the smallest term to generate, used
    to minimize computation must be less than term\_limit (default is 2)
  - step\_len - (Optional) size of steps to summarize the distribution
    (between 0.001 and 0.01, which is between approx 1000 and 100
    summarized points). This is only used if the data vector length is
    greater than 100.
  - probs - (Optional) probability quantiles, same length as x

Here is an example of a lower bounded distribution build.

``` r
my_metalog <- metalog(
  fishSize$FishSize,
  term_limit = 9,
  term_lower_bound = 2,
  bounds = c(0, 60),
  boundedness = 'b',
  step_len = 0.01
  )
```

The function returns an object of class `rmetalog` and `list`. You can
get a summary of the distributions using `summary`.

``` r
summary(my_metalog)
#>  -----------------------------------------------
#>  Summary of Metalog Distribution Object
#>  -----------------------------------------------
#>  
#> Parameters
#>  Term Limit:  9 
#>  Term Lower Bound:  2 
#>  Boundedness:  b 
#>  Bounds (only used based on boundedness):  0 60 
#>  Step Length for Distribution Summary:  0.01 
#>  Method Use for Fitting:  any 
#>  
#> 
#>  Validation and Fit Method
#>  term valid method
#>     2   yes    OLS
#>     3   yes    OLS
#>     4   yes    OLS
#>     5   yes    OLS
#>     6   yes    OLS
#>     7   yes    OLS
#>     8   yes    OLS
#>     9   yes    OLS
```

You can also plot a quick visual comparison of the distributions by
term.

``` r
plot(my_metalog)
#> $pdf
```

![](man/figures/README-unnamed-chunk-8-1.png)<!-- -->

    #> 
    #> $cdf

![](man/figures/README-unnamed-chunk-8-2.png)<!-- -->

Once the distributions are built, you can create `n` samples by
selecting a term.

``` r
s <- rmetalog(my_metalog, n = 1000, term = 9)
hist(s)
```

![](man/figures/README-unnamed-chunk-9-1.png)<!-- -->

You can also retrieve quantile, density, and probability values similar
to other R distributions.

``` r
qmetalog(my_metalog, y = c(0.25, 0.5, 0.75), term = 9)
#> [1]  7.241  9.840 12.063
```

probabilities from a quantile.

``` r
pmetalog(my_metalog, q = c(3, 10, 25), term = 9)
#> [1] 0.001957 0.520058 0.992267
```

density from a quantile.

``` r
dmetalog(my_metalog, q = c(3, 10, 25), term = 9)
#> [1] 0.004490 0.126724 0.002264
```

As this package is under development, any feedback is appreciated\!
Please submit a pull request or issue if you find anything that needs to
be addressed.
