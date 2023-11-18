from kpsn.util import keypt_io
import seaborn as sns

age_pal_from_unique = lambda age_keys: dict(zip(
    sorted(age_keys),
    sns.hls_palette(l = 0.4, n_colors = len(age_keys) + 2)[1:-1])
)
age_pal = lambda ages: age_pal_from_unique(
    keypt_io.get_groups_dict(ages)[0])