from kpsn.util import keypt_io
import seaborn as sns

hls_pal = lambda n: sns.hls_palette(l = 0.4, n_colors = n + 2)[1:-1]
autumn_pal = lambda n: sns.color_palette('autumn', n_colors = n)

age_pal_from_unique = lambda age_keys, func: dict(zip(
    sorted(age_keys),
    func(len(age_keys))
))
age_pal = lambda ages, func=hls_pal: age_pal_from_unique(
    keypt_io.get_groups_dict(ages)[0], func)