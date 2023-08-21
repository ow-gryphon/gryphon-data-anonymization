import pandas as pd
import numpy as np

from cryptography.fernet import Fernet
import itertools
import random
import string


class DataAnonymization:
    """DataAnonymization is class that contains the basic transformations to anonymize a Pandas DataFrame.

    Args:
        data (Pandas DF): Pandas DF that will be transformed.

    Attributes:
        data (Pandas DF): Pandas DF that will be transformed.
        map_cat_dict (Dictionary): Backup of the categories mapping applied to each column.
        key_dict (Dictionary): Backup of the key used to encrypt each column.
        report_list (List): List of strings with the transformations applied.

    """

    def __init__(self, data=None):
        self.data = data.copy()
        self.map_cat_dict = {}
        self.key_dict = {}
        self.report_list = []

    @property
    def report(self):
        """Report that summarizes the transformation applied by each instance of the class."""
        
        try:
            from IPython.display import Markdown, display
            display(Markdown("**-- REPORT --**"))
            [
                display(Markdown(f"**{cont + 1} - {self.report_list[cont]}"))
                for cont in range(len(self.report_list))
            ]
        except:
            print("**-- REPORT --**")
            [
                print(f"**{cont + 1} - {self.report_list[cont]}")
                for cont in range(len(self.report_list))
            ]
        

    # GENERIC
    # Suppression
    def suppression(self, col, fill_value=None):
        """Replace information in a column.

        Args:
            col (string): Column target of the transformation.
            fill_value (string / float / int): Value set in the target column (Default value: None).

        """
        self.report_list += [f"Suppression:** col={col} fill_value={fill_value}"]
        self.data[col] = fill_value

    # Shuffling
    def shuffling(self, col):
        """Randomly shuffle the values of a column.

        Args:
            col (string): Column target of the transformation.

        """
        self.report_list += [f"Shuffling:** col={col}"]
        x = np.array(self.data[col])
        np.random.shuffle(x)
        self.data[col] = x

    # NUMERICAL
    
    # Replace with Gaussian
    def replace_gaussian(self, col, a_min=-np.inf, a_max=np.inf):
        """Replace with normal (Gaussian) distribution.

        Args:
            col (string): Column target of the transformation.
            a_min (float): Minimum value allowed in the column (Default value: -inf).
            a_max (float): Maximum value allowed in the column (Default value: +inf).

        """

        self.report_list += [
            f"Replace with normal distribution:** col={col} a_min={a_min} a_max={a_max}"
        ]
        x = np.array(self.data[col])
        std = np.nanstd(x, axis=0)
        mean = np.nanmean(x, axis=0)
        self.data[col] = np.clip(
            np.random.normal(loc=mean, scale=std, size=x.shape),
            a_min,
            a_max,
        )
    
    
    # Adding Noise
    def noise_num(self, col, ratio_std=0.1, a_min=-np.inf, a_max=np.inf):
        """Add random noise from a normal (Gaussian) distribution.

        Args:
            col (string): Column target of the transformation.
            ratio_std (float): Ratio of the column standard deviation used to define the noise standard deviation (Default value: 0.1).
            a_min (float): Minimum value allowed in the column (Default value: -inf).
            a_max (float): Maximum value allowed in the column (Default value: +inf).

        """

        self.report_list += [
            f"Noise numeric:** col={col} ratio_std={ratio_std} a_min={a_min} a_max={a_max}"
        ]
        x = np.array(self.data[col])
        std = np.nanstd(x, axis=0)
        self.data[col] = x + np.clip(
            np.random.normal(loc=0, scale=ratio_std * std, size=x.shape),
            a_min,
            a_max,
        )

    # Bucket
    def bucket_num(self, col, a_min, a_max, a_step):
        """Bucketize numerical column.

        Args:
            col (string): Column target of the transformation.
            a_min (float): Lowest threshold of the buckets.
            a_max (float): Highest threshold of the buckets.
            a_step (float): Buckets range.

        """

        self.report_list += [
            f"Bucket numeric:** col={col} a_min={a_min} a_max={a_max} a_step={a_step}"
        ]
        x = self.data[col]
        list_bins = [-np.inf] + list(np.arange(a_min, a_max, a_step)) + [np.inf]
        self.data[col] = pd.cut(x, bins=list_bins, right=False).astype(str)

    # Cap Outliers
    def cap_outliers(self, col, q_min=0.05, q_max=0.95):
        """Cap values higher/lower of a certain quantile.

        Args:
            col (string): Column target of the transformation.
            q_min (float): Minimum quantile (Default value: 0.05).
            q_max (float): Maximum quantile (Default value: 0.95).

        """
        self.report_list += [f"Cap Outliers:** col={col} q_min={q_min} q_max={q_max}"]
        x = self.data[col]
        v_low = np.nanquantile(x, q_min, axis=0)
        v_high = np.nanquantile(x, q_max, axis=0)
        self.data[col] = np.clip(x, v_low, v_high)

    # DATE
    # Shift Dates
    def noise_date(self, col, low, high, delta_type="D"):
        """Add random uniform noise to dates, it can be days / months / years.

        Args:
            col (string): Column target of the transformation.
            low (float): Minimum value of the uniform distribution.
            q_max (float): Maximum value of the uniform distribution.
            delta_type (string): Date / time unit in which the noise is added (days ('D'), months ('M'), years ('Y')) (Devault value: 'D').

        """

        self.report_list += [
            f"Noise date:** col={col} low={low} high={high} delta_type={delta_type}"
        ]

        def f(x, delta):
            return np.datetime64(x, delta_type) + np.timedelta64(delta, delta_type)

        delta = np.random.randint(low=low, high=high, size=len(self.data[col]))
        self.data[col] = np.vectorize(f)(self.data[col], delta)

    # CATEGORICAL
    # Hash
    def hash_round(self, col, factor=1, hash_key='0123456789123456'):
        """Hash the values of a column. The output value can be divided by a factor to difficult reverse engineering, although it increase the probability of collisions.

        Args:
            col (string): Column target of the transformation.
            factor (float): Factor by which the hash value is divided and rounded (Default value: 1).

        """

        self.report_list += [f"Hash:** col={col} factor={factor}"]
        x = np.array(self.data[col])
        self.data[col] = (np.around(pd.util.hash_array(x, hash_key=hash_key) / factor, 0)).astype("uint64")

    # Encrypt
    def encrypt_fernet(self, col, key):
        """Encrypt a column with a user-defined key.

        Args:
            col (string): Column target of the transformation.
            key (string): Key used to encrypt / decrypt the information.

        """

        self.report_list += [f"Encrypt:** col={col} key={key}"]

        def f(x, key):
            return Fernet(key).encrypt(str(x).encode())

        self.key_dict[col] = key
        self.data[col] = np.vectorize(f)(np.array(self.data[col]), key)

    def decrypt_fernet(self, col, key=None):
        """Decrypt a column with a user-defined key.

        Args:
            col (string): Column target of the transformation.
            key (string): Key used to encrypt / decrypt the information.

        """

        def f(x, key):
            return Fernet(key).decrypt(x).decode()

        if key is None:
            key = self.key_dict[col]
        self.report_list += [f"Decrypt:** col={col} key={key}"]
        self.data[col] = np.vectorize(f)(np.array(self.data[col]), key)

    # Encode
    # https://stackoverflow.com/questions/42176498/repeating-letters-like-excel-columns
    def excel_cols(self):
        """Create a list of alphabetic columns IDs, equivalent to excel format."""

        n = 1
        while True:
            yield from (
                "".join(group)
                for group in itertools.product(string.ascii_uppercase, repeat=n)
            )
            n += 1

    def map_cat(self, col, map_type=1, map_cat_dict=None):
        """Codify categorical columns with new identifiers that can be letters (1) or numbers (2).

        Args:
            col (string): Column target of the transformation.
            map_type (integer): 1 -> Map to alphabetic coding.
                                2 -> Map to numeric coding.

        """

        self.report_list += [f"Mapping categories:** col={col} map_type={map_type}"]
        x = self.data[col]
        x_unique = np.unique(x)
        n_codes = len(x_unique)

        if map_type == 1:
            list_codes = list(itertools.islice(self.excel_cols(), n_codes))
        else:
            list_codes = list(range(1, n_codes + 1))
        random.shuffle(list_codes)

        # Extract out keys and values
        v = np.array(x_unique)
        c = np.array(list_codes)

        if map_cat_dict is not None:
            v_miss = np.array([item for item in v if item not in map_cat_dict["value"]])
            c_miss = np.array([item for item in c if item not in map_cat_dict["code"]])
            v = map_cat_dict["value"]
            c = map_cat_dict["code"]
            if len(v_miss) > 0:
                v = np.concatenate([v, v_miss], axis=0)
                c = np.concatenate([c, c_miss], axis=0)
        
        self.map_cat_dict[col] = {"value": v, "code": c}

        # Get argsort indices
        sidx = v.argsort()

        self.data[col] = c[sidx[np.searchsorted(v, x, sorter=sidx)]]

    # Remove Small Categories
    def remove_small_cats(self, col, factor, fill_type="most_frequent"):
        """Remove categories that are underrepresented.

        Args:
            col (string): Column target of the transformation.
            factor (float / integer): Factor used to define what is considered a category with low representation:
                                      < 1 -> Minimum percentage of rows that contain a specific category (float).
                                      >= 1-> Minimum number of samples that have a specific category (Integer).
            fill_type (string): Type of replacement value replacement (Default value: most_frequent):
                                most_frequent -> Fill with the most frequent category.
                                rand -> Fill with another random category with a probability according to its distribution in the dataset.

        """

        self.report_list += [
            f"Remove small categories:** col={col} factor={factor} fill_type={fill_type}"
        ]
        x = self.data[col].copy()
        unique, counts = np.unique(x, return_counts=True)

        if factor < 1:
            factor = factor * len(x)

        criteria = np.argwhere(np.isin(x, unique[counts <= factor])).ravel()

        if fill_type == "most_frequent":
            mode = unique[np.argmax(counts)]
            x[criteria] = mode

        elif fill_type == "rand":
            x[criteria] = np.random.choice(
                unique[counts > factor],
                len(x[criteria]),
                p=counts[counts > factor] / sum(counts[counts > factor]),
            )
        else:
            raise ValueError("fill_type must be either 'most_frequent' or 'rand'")

        self.data[col] = x

    # DATASET
    # Sampling
    def sampling(self, sample):
        """Random sampling of the dataset.

        Args:
            sample (float / integer): < 1 -> Percentage of rows sampled (float).
                                      >= 1-> Number of rows sampled (Integer).

        """

        self.report_list += [f"Sampling:** sample={sample}"]
        if sample >= 1:
            self.data = self.data.sample(n=sample, replace=False)
        else:
            self.data = self.data.sample(frac=sample, replace=False)