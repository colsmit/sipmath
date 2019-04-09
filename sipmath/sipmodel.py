import numpy as np
import pandas as pd
import random
import json
import xml.etree.ElementTree as ET

from . import sipinput

class sipmodel:

    def __init__(self, trials, **kwargs):
        self.inputs = []
        self.trials = int(trials)
        self.apply_params(kwargs)
        self.extra_dims = 0

    def apply_params(self, params):
        param_defaults = {
            # norm
            'name': '',
            'coherent': '',
            'count': str(self.trials),
            'about': '',
            'origin': '',
            'dataver': ''

        }

        for (param, default) in param_defaults.items():
            setattr(self, param, params.get(param, default))

    def sipinput(self, dims=None, distribution=None, **kwargs):
        v_ind_seed = random.randint(0, 8000000)
        v_ind = len(self.inputs) + v_ind_seed
        a_ind = len(self.inputs)

        if type(dims) == int and dims > 1:
            n_si = sipinput(shape=(self.trials, dims), distribution=distribution, v_ind=v_ind, a_ind=a_ind, parent=None, **kwargs)
            self.extra_dims += dims - 1
        else:
            n_si = sipinput.sipinput(shape=(self.trials), distribution=distribution, v_ind=v_ind, a_ind=a_ind, parent=None, **kwargs)

        if "correlated" in distribution:
            n_sis = []
            for i in n_si.mean:
                # add parent
                n_si_c = sipinput.sipinput(shape=(self.trials), distribution=distribution, v_ind=v_ind, a_ind=a_ind, parent=n_si, **kwargs)
                self.inputs.append(n_si_c)
                n_sis.append(n_si_c)
                v_ind += 1
                a_ind += 1
            return tuple(n_sis)

        else:
            self.inputs.append(n_si)
            return n_si

    def sample(self):
        ydim = self.trials
        xdim = len(self.inputs)

        out = np.empty((ydim, xdim + self.extra_dims))
        j = 0
        for i in range(xdim):
            if j > 0:
                j -= 1
                continue

            col_idx = i

            vals = self.inputs[i].generate_samples(ydim)

            j = i + np.shape(vals)[1]
            out[:, i:j] = vals
            for k in range(i, j):
                ipts = self.inputs
                if np.size(ipts[k][:]) > ydim:
                    ipts[k][:] = out[:, k:(k + ipts[k].shape[1])]
                    break
                else:
                    ipts[k][:] = out[:, k].T
                self.inputs = ipts
            j -= 1 + i
        self.gens = out

    def to_df(self, labels=False):
        ls = []
        for ipt in self.inputs:
            if ipt.dims > 1:
                for i in range(1, ipt.dims + 1):
                    ls.append(str(ipt.name) + '_' + str(i))
            else:
                ls.append(ipt.name)

        df = pd.DataFrame(self.gens)
        df.columns = ls
        df.index.name = 'trial'
        return df

    def from_df(self, df):
        # exceptions: different length sips
        n_sis = []
        for col in df:
            n_si = self.sipinput(distribution='from_df', name=col)
            n_sis.append(n_si)

        ydim = self.trials
        xdim = len(self.inputs)

        if xdim != len(n_sis):
            # attempted to instandiate non-empty sipmodel
            raise ValueError('Must import DataFrame into empty Sipmodel.')

        out = np.empty((xdim, ydim))
        for i in range(xdim):
            vals = df.iloc[:, i].values

            out[i, :] = vals.T
            ipts = self.inputs
            ipts[i][:] = out[i, :]
            self.inputs = ipts
        self.gens = out

        return tuple(n_sis)

    def from_sip(self, path):
        # error handling for xmlpath/xmltype
        etree = ET.parse(path)
        cols = [str(json.dumps(elem.attrib)) for elem in etree.iter() if elem.tag != 'SLURP']

        df = pd.DataFrame(columns=cols)

        for elt in etree.iter():
            if elt.tag == 'SLURP':
                self.apply_params(elt.attrib)
            if elt.tag != 'SLURP':
                df[str(json.dumps(elt.attrib))] = elt.text.split(',')
        df.convert_objects(convert_numeric=True)
        return self.from_df(df)

    def to_xml(self, path):
        # error handling: path is actually a path
        slurp_element = ET.Element('SLURP')
        slurp_element.attrib = self.get_xmlattrib()

        for sip in self.inputs:
            sip_subelement = ET.SubElement(slurp_element, 'SIP')
            # need method for this
            sip_subelement.attrib = sip.get_xmlattrib()
            # placeholder = row of numpy subarray w/gens
            sip_subelement.text = ','.join([str(num) for num in sip])
        try:
            outxmlfile = open(path, 'wb')
        except OSError:
            raise OSError('Error writing to file.')
        outxmlfile.write(ET.tostring(slurp_element))
        outxmlfile.close()

    def get_xmlattrib(self):
        # error handling: if model hasnt been sampled?
        # gotta make sure all params are actually in sipinputs lol
        params = ["name", "coherent", "count", "about", "origin", "dataver"]
        attribsdict = {}

        for param in params:
            attrib = str(getattr(self, param))
            if len(attrib) > 0:
                attribsdict[param] = attrib

        return attribsdict