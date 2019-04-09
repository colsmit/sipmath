import scipy.stats as sp
import numpy as np
import json
import sipmath.pymetalog as pm

class sipinput(np.ndarray):
    """
        The sipinput class denotes a single or multidimensional SIP that belongs to a larger SLURP or sipmodel.
        A sipinput inherits from a numpy ndarray as it is a 2d or 3d array of samples from one (in the 2d
        case) or many (in the 3d case) distributions. This inheritence allows for the use of ndarray arithmatic methods
        directly on sipinputs.
        ----------
         shape : integer or tuple
            Shape of sipinput, if 1d this shape n is an integer s.t. sipinput
            shape = nx1. If tuple, shape is (n,m) s.t. shape = nxm. This value
            is produced from sipmodel instantiation, n is usually the number of
            sipmodel trials.
         distribution : string
            Distribution that sipinnput values are sampled from. Must be one of
            the supported scipy distributions or metalog.
         v_ind : int
            V_ind is used as a seed for hdr generation and is inherited from
            sipmodel instantiation.
         a_ind : int
            Used for naming sipinput if no name is suspplied. Also the index of
            the sipmodel in sipmodel.inputs.
        parent : sipinput or None
            Sipinputs with multivariate distributions often have multiple columns
            sampled from different, similar distributions. These cases have
            been handled by creating a parent-child relationship where a parent
            generates samples from multivariate distributions and gives them
            to its children.
        name : string, optional
            Name of the sipinput, shows up in df formulation of sipmodel and
            SIP metadata in XML format.
        **kwargs : string, int, float, array, optional
            Keyword arguments are passed either for scipy distribution parameters
            or XML metadata.

        Methods
        -------
        __new__ : Instantiates and returns instance of sipinput class
        apply_params : Sets/validates distribution and metadata parameters from
            **kwargs
        random_trials : Returns uniform distribution for use in distribution
            sampling. Set by keyword argument to either native numpy generator
            or hdr.
        get_xmlattrib : Returns list of XML metadata parameters for SIP when
            generating XML file of sipmodel.
        generate_samples : Returns  an array of samples from distribution during
            sipmodel.sample().


    """
    # class instantiation
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0, strides=None, order=None, distribution=None, v_ind=0,
                a_ind=0, parent=None, name=None, **kwargs):
        obj = np.ndarray.__new__(subtype, shape, dtype, buffer, offset, strides, order)

        obj.distribution = distribution
        obj.v_ind = v_ind
        obj.a_ind = a_ind
        obj.parent = parent
        obj.shape = shape

        if type(shape) == tuple:
            obj.dims = shape[1]
        else:
            obj.dims = 1

        # name setting
        if name != None:
            if '{"' in name and '"}' in name:
                n_kwargs = json.loads(name)
                obj.apply_params(n_kwargs)
            else:
                obj.name = name
        else:
            name = "var_" + str(a_ind + 1)
            obj.name = name

        obj.apply_params(kwargs)

        return obj

    def apply_params(self,  params):

        distribution_parameters = {
            'uniform':{'loc':0, 'scale':1},
            'normal':{'loc':0, 'scale':1},
            'beta': {'a':0, 'b':0,'loc':0, 'scale':1},
            'binomial': {'n':0, 'p':0,'loc':0},
            'chisquared': {'df':0, 'loc':0, 'scale':1},
            'exponential': {'loc':0, 'scale':1},
            'f': {'dfn':0, 'dfd':0,'loc':0, 'scale':1},
            'discrete': {'xk':[0], 'pk':[1]},
            'gamma': {'a':0,'loc':0, 'scale':1},
            'lognormal': {'s':0, 'loc':0, 'scale':1},
            'poisson': {'mu':0, 'loc':0},
            'triangular': {'c':0, 'loc':0, 'scale':1},
            't': {'mu':0, 'loc':0, 'scale':1},
            'weibull_min': {'c':0, 'loc':0, 'scale':1},
            'correlated_normal': {'mean':[0], 'cov':[0]},
            'correlated_uniform': {'mean':[0], 'cov':[0]},
            'metalog': {'metalog':None, 'term':1},
            'from_df': {}
        }

        # set attributes unique ot distribution input
        # this line is why this package requires python 3+ (.items())
        for (param, default) in distribution_parameters[self.distribution].items():
            setattr(self, param, params.get(param, default))

            # keep track of updated params
            if param in params:
                del params[param]

        sip_metadata_parameters = {
            'origin': '',
            'csvr': '',
            'copyright': '',
            'dataver': '',
            'dims': 0,
            'offset': '',
            'provenance': '',
            'units': '',
            'ver': '',
            'count': '',
            'name': None,
            'type': '',
            'min': 0,
            'max': 0,
            'avg': 0,
            'about': '',
            'generator':'rand'
        }

        for (param, default) in sip_metadata_parameters.items():
            setattr(self, param, params.get(param, default))

            if param in params:
                del params[param]

        if len(params) > 0:
            raise TypeError('Unexpected Keyword Argument(s): {}'.format(", ".join(params)))

    def random_trials(self):
        # for generating uniform random values [0,1] to feed into distribution inverse survival functions

        size = np.prod(self.shape)

        # hdr random number generator functionality
        if self.generator == "hdr":
            x = np.arange(1, size + 1)
            v_index = self.v_ind

            def hdrgen(pm_index):
                return (np.mod(((np.mod((v_index + 1000000) ^ 2 + (v_index + 1000000) * (pm_index + 10000000),
                                        99999989)) + 1000007) * ((np.mod(
                    (pm_index + 10000000) ^ 2 + (pm_index + 10000000) * (
                        np.mod((v_index + 1000000) ^ 2 + (v_index + 1000000) * (pm_index + 10000000), 99999989)),
                    99999989)) + 1000013), 2147483647) + 0.5) / 2147483647

            vhdrgen = np.vectorize(hdrgen)
            return vhdrgen(x)

        # else use standard numpy random number generator
        else:
            return np.random.rand(size)

    def get_xmlattrib(self):
        # error handling: if model hasnt been sampled?
        params = ["name", "count", "type", "min", "max", "avg", "about", "origin", "ver", "csvr", "copyright",
                  "dataver", "dims", "offset", "origin", "provenance", "units"]
        attribsdict = {}

        for param in params:
            attrib = str(getattr(self, param))
            if len(attrib) > 0:
                if param == 'mean':
                    attribsdict.update({'sip_mean': attrib})
                else:
                    attribsdict.update({param: attrib})

        return attribsdict

    def generate_samples(self, trials):

        #TODO might be an error here with child SIPinput recieving all of parent inputs
        if type(self.parent) != type(None):
            return self.parent.generate_samples(trials)

        rt = self.random_trials()

        d = self.distribution

        if d == 'from_df':
            raise TypeError("Cannot generate samples for imported SIPs")

        if d == 'uniform':
            # isf(q, loc=0, scale=1)
            out = sp.uniform.isf(rt, self.loc, self.scale)

        if d == 'normal':
            # isf(q, loc=0, scale=1) loc=mean scale=stdev
            out = sp.norm.isf(rt, self.loc, self.scale)

        if d == 'beta':
            # isf(q, a, b, loc=0, scale=1) alpha, beta, a, b
            out = sp.beta.isf(rt, self.a, self.b, self.loc, self.scale)

        if d == 'binomial':
            out = sp.binom.isf(rt, self.n, self.p, self.loc)

        if d == 'chisquared':
            # isf(q, df, loc=0, scale=1)
            out = sp.chi2.isf(rt, self.df, self.loc, self.scale)

        if d == 'exponential':
            # isf(q, loc=0, scale=1)
            out = sp.expon.isf(rt, self.loc, self.scale)

        if d == 'f':
            # isf(q, dfn, dfd, loc=0, scale=1)
            out = sp.ncf.isf(rt, self.dfn, self.dfd, self.loc, self.scale)

        if d == 'discrete':
            # gotta add some values input validation
            rd = sp.rv_discrete(values=(self.xk, self.pk))
            out = rd.isf(rt)

        if d == 'gamma':
            # isf(q, a, loc=0, scale=1)
            out = sp.gamma.isf(rt, self.a, self.loc, self.scale)

        if d == 'lognormal':
            # isf(q, s, loc=0, scale=1)
            out = sp.lognorm.isf(rt, self.s, self.loc, self.scale)

        if d == 'poisson':
            # isf(q, mu, loc=0)
            out = sp.poisson.isf(rt, self.mu, self.loc)

        if d == 'triangular':
            # isf(q, c, loc=0, scale=1)
            out = sp.triang.isf(rt, self.c, self.loc, self.scale)

        if d == 't':
            # isf(q, df, loc=0, scale=1)
            out = sp.poisson.isf(rt, self.mu, self.loc, self.scale)

        if d == 'weibull_min':
            out = sp.weibull_min.isf(rt, self.c, self.loc, self.scale)

        if d == 'correlated_normal':
            out = sp.multivariate_normal.rvs(mean=self.mean, cov=self.cov, size=trials, random_state=None)

        if d == 'correlated_uniform':
            c_norm = sp.multivariate_normal.rvs(mean=self.mean, cov=self.cov, size=trials, random_state=None)
            out = np.zeros_like(c_norm)
            for col in range(np.shape(out)[1]):
                out[:, col] = sp.norm.sf(c_norm[:, col], loc=self.mean[col], scale=np.std(c_norm[:, col]))

        if d == 'metalog':
            out = pm.rmetalog(self.metalog, n=trials, term=self.term)

        # for multidimensional sips (sips with more than one column)
        out = np.reshape(out, (trials, np.size(out) // trials))

        # set metadata params
        self.max = np.max(out)
        self.min = np.min(out)
        self.avg = np.mean(out)
        return out

    # input validation...
    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, distributions):
        dists = ['uniform','normal','beta','binomial','chisquared','exponential','f','discrete','gamma','lognormal',
                 'poisson','triangular','t','weibull_min','correlated_normal','correlated_uniform','metalog', 'from_df']
        if (type(distributions) != str):
            raise TypeError('Distribution parameter must be a string')
        if distributions not in dists:
            raise ValueError('Distribution input is either not a valid input or not supported, valid distribution'+
                             ' inputs are: {}'.format(" ".join(dists)))
        self._distribution = distributions

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, locs):
        if (type(locs) != int):
            raise TypeError('Distribution parameter loc must be an integer')
        self._loc = locs

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scales):
        if (type(scales) != int):
            raise TypeError('Distribution parameter scale must be an integer')
        self._scale = scales

    @property
    def stdev(self):
        return self._stdev

    @scale.setter
    def stdev(self, stdevs):
        if (type(stdevs) != int):
            raise TypeError('Distribution parameter stdev must be an integer')
        self._stdev = stdevs

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a_s):
        if (type(a_s) != int):
            raise TypeError('Distribution parameter a must be an integer')
        self._scale = a_s

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, ns):
        if (type(ns) != int):
            raise TypeError('Distribution parameter n must be an integer')
        self._n = ns

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, bs):
        if (type(bs) != int):
            raise TypeError('Distribution parameter b must be an integer')
        self._b = bs

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, ps):
        if (type(ps) != int):
            raise TypeError('Distribution parameter p must be an integer')
        self._p = ps

    @property
    def s(self):
        return self._s

    @s.setter
    def s(self, ss):
        if (type(ss) != int):
            raise TypeError('Distribution parameter s must be an integer')
        self._s = ss

    @property
    def dfn(self):
        return self._dfn

    @dfn.setter
    def dfn(self, dfns):
        if (type(dfns) != int):
            raise TypeError('Distribution parameter dfn must be an integer')
        self._dfn = dfns

    @property
    def dfd(self):
        return self._dfd

    @dfd.setter
    def dfd(self, dfds):
        if (type(dfds) != int):
            raise TypeError('Distribution parameter dfd must be an integer')
        self._dfd = dfds

    @property
    def generator(self):
        return self._generator

    @generator.setter
    def generator(self, generators):
        if (type(generators) != str):
            raise TypeError('Distribution generator must be a string')
        if generators not in ['rand', 'hdr']:
            raise ValueError('Distribution generator must be \'rand\' or \'hdr\'')
        self._generator = generators

    @property
    def xk(self):
        return self._xk

    @xk.setter
    def xk(self, xks):
        if (type(xks) != 'list'):
            raise TypeError('Distribution parameter xk must be a list')
        for xs in xks:
            if (type(xs) != int):
                raise TypeError('Distribution parameter xk must be a list containing integers')
        self._xk = xks

    @property
    def pk(self):
        return self._pk

    @pk.setter
    def pk(self, pks):
        if (type(pks) != 'list'):
            raise TypeError('Distribution parameter pk must be a list')
        for ps in pks:
            if (type(ps) != int):
                raise TypeError('Distribution parameter pk must be a list containing integers')
        self._pk = pks

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, mus):
        if (type(mus) != int):
            raise TypeError('Distribution parameter mu must be an integer')
        self._mu = mus

    @property
    def c(self):
        return self._c

    @mu.setter
    def c(self, cs):
        if (type(cs) != int):
            raise TypeError('Distribution parameter c must be an integer')
        self._c = cs

    @property
    def sip_mean(self):
        return self._sip_mean

    @sip_mean.setter
    def sip_mean(self, sip_means):
        if (type(sip_means) != 'list'):
            raise TypeError('Distribution parameter mean must be a list')
        for ms in sip_means:
            if (type(ms) != int):
                raise TypeError('Distribution parameter mean must be a list containing integers')
        self._sip_mean = sip_means

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, covs):
        if (type(covs) != 'list'):
            raise TypeError('Distribution parameter cov must be a list')
        for cs in covs:
            if (type(cs) != int):
                raise TypeError('Distribution parameter cov must be a list containing integers')
        self._cov = covs

    @property
    def origin(self):
        return self._origin

    @origin.setter
    def origin(self, origins):
        if (type(origins) != str):
            raise TypeError('SIP origin metadatamust be a string')
        self._origin = origins

    @property
    def csvr(self):
        return self._csvr

    @csvr.setter
    def csvr(self, csvrs):
        if (type(csvrs) != str):
            raise TypeError('SIP csvr metadatamust be a string')
        self._csvr = csvrs

    @property
    def copyright(self):
        return self._copyright

    @copyright.setter
    def copyright(self, copyrights):
        if (type(copyrights) != str):
            raise TypeError('SIP copyright metadatamust be a string')
        self._copyright = copyrights

    @property
    def dataver(self):
        return self._dataver

    @dataver.setter
    def dataver(self, datavers):
        if (type(datavers) != str):
            raise TypeError('SIP dataver metadata must be a string')
        self._dataver = datavers

    @property
    def dims(self):
        return self._dims

    @dims.setter
    def dims(self, dimss):
        try:
            dimss = int(dimss)
        except:
            raise TypeError('SIP dims metadata must be an integer')
        if (type(dimss) != int):
            raise TypeError('SIP dims metadata must be an integer')
        self._dims = dimss

    @property
    def offset(self):
        return self._offset

    @offset.setter
    def offset(self, offsets):
        if (type(offsets) != str):
            raise TypeError('SIP offset metadata must be a string')
        self._offset = offsets

    @property
    def provenance(self):
        return self._provenance

    @provenance.setter
    def provenance(self, provenances):
        if (type(provenances) != str):
            raise TypeError('SIP provenance metadata must be a string')
        self._provenance = provenances

    @property
    def units(self):
        return self._units

    @units.setter
    def units(self, unitss):
        if (type(unitss) != str):
            raise TypeError('SIP units metadata must be a string')
        self._units = unitss

    @property
    def ver(self):
        return self._ver

    @ver.setter
    def ver(self, vers):
        if (type(vers) != str):
            raise TypeError('SIP units metadata must be a string')
        self._ver = vers

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, counts):
        self._count = counts

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, names):
        if names == None:
            self._name = self.name
        elif (type(names) != str):
            raise TypeError('SIP name metadata must be a string')
        else:
            self._name = names

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, types):
        if (type(types) != str):
            raise TypeError('SIP type metadata must be a string')
        self._type = types

    @property
    def min(self):
        return self._min

    @min.setter
    def min(self, mins):
        try:
            mins = float(mins)
        except:
            raise TypeError('SIP min metadata must be numeric')
        self._min = mins

    @property
    def max(self):
        return self._max

    @max.setter
    def max(self, maxs):
        try:
            maxs = float(maxs)
        except:
            raise TypeError('SIP max metadata must be numeric')
        self._max = maxs

    @property
    def avg(self):
        return self._avg

    @avg.setter
    def avg(self, avgs):
        try:
            avgs = float(avgs)
        except:
            raise TypeError('SIP avg metadata must be numeric')
        self._avg = avgs

    @property
    def about(self):
        return self._about

    @about.setter
    def about(self, abouts):
        if (type(abouts) != str):
            raise TypeError('SIP about metadata must be a string')
        self._about = abouts

    @property
    def metalog(self):
        return self._metalog

    @metalog.setter
    def metalog(self, metalogs):
        #note, this will absolutely break, figure out wtf to do
        print(type(metalogs))
        if (type(metalogs) != pm.metalog):
            raise TypeError('Metalog distribution must be of type metalog')
        self._metalog = metalogs

    @property
    def term(self):
        return self._term

    @term.setter
    def term(self, terms):
        if (type(terms) != int):
            raise TypeError('SIP term metadata must be an integer')
        self._term = terms

