#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_cell_magic('bash', '', 'git clone https://github.com/chetu26/farm-field.git\n')


# In[ ]:


# %load farm-field/utils/img.py
#!/usr/bin/env python

# In[1]:


# img.py

from typing import Tuple, Generator, List, Union

import rasterio.windows
from rasterio.windows import Window
from shapely.geometry import Polygon
import affine
import warnings
from pathlib import Path

import itertools
import numpy as np
import rasterio
import shapely
from shapely.geometry import Polygon
from PIL import Image as pilimg
from skimage import exposure, img_as_ubyte
from tqdm import tqdm


def get_chip_windows(raster_width: int,
                     raster_height: int,
                     raster_transform,
                     chip_width: int=256,
                     chip_height: int=256,
                     skip_partial_chips: bool=False,
                     ) -> Generator[Tuple[Window, affine.Affine, Polygon], any, None]:
    """Generator for rasterio windows of specified pixel size to iterate over an image in chips.
    Chips are created row wise, from top to bottom of the raster.
    Args:
        raster_width: rasterio meta['width']
        raster_height: rasterio meta['height']
        raster_transform: rasterio meta['transform']
        chip_width: Desired pixel width.
        chip_height: Desired pixel height.
        skip_partial_chips: Skip image chips at the edge of the raster that do not result in a full size chip.
    Returns :
        Yields tuple of rasterio chip window, chip transform and chip polygon.
    """
    col_row_offsets = itertools.product(range(0, raster_width, chip_width), range(0, raster_height, chip_height))
    raster_window = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)

    for col_off, row_off in col_row_offsets:
        chip_window = Window(col_off=col_off, row_off=row_off, width=chip_width, height=chip_height)

        if skip_partial_chips:
            if row_off + chip_height > raster_height or col_off + chip_width > raster_width:
                continue

        chip_window = chip_window.intersection(raster_window)
        chip_transform = rasterio.windows.transform(chip_window, raster_transform)
        chip_bounds = rasterio.windows.bounds(chip_window, raster_transform)  # Uses transform of full raster.
        chip_poly = shapely.geometry.box(*chip_bounds, ccw=False)

        yield (chip_window, chip_transform, chip_poly)


def cut_chip_images(inpath_raster: Union[Path, str],
                    outpath_chipfolder: Union[Path, str],
                    chip_names: List[str],
                    chip_windows: List,
                    bands=[3, 2, 1]):
    """Cuts image raster to chips via the given windows and exports them to jpg."""

    src = rasterio.open(inpath_raster)

    all_chip_stats = {}
    for chip_name, chip_window in tqdm(zip(chip_names, chip_windows)):
        img_array = np.dstack(list(src.read(bands, window=chip_window)))
        img_array = exposure.rescale_intensity(img_array, in_range=(0, 2200))  # Sentinel2 range.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_array = img_as_ubyte(img_array)
        img_pil = pilimg.fromarray(img_array)

        # Export chip images
        Path(outpath_chipfolder).mkdir(parents=True, exist_ok=True)
        with open(Path(rf'{outpath_chipfolder}\{chip_name}.jpg'), 'w') as dst:
            img_pil.save(dst, format='JPEG', subsampling=0, quality=100)

        all_chip_stats[chip_name] = {'mean': img_array.mean(axis=(0, 1)),
                                     'std': img_array.std(axis=(0, 1))}
    src.close()

    return all_chip_stats


# In[ ]:






# In[ ]:


# %load farm-field/utils/geo.py
#!/usr/bin/env python

# In[2]:


get_ipython().run_cell_magic('bash', '', 'git clone https://github.com/chetu26/farm-field.git\n')


# In[ ]:


# %load farm-field/utils/img.py
#!/usr/bin/env python

# In[1]:


# img.py

from typing import Tuple, Generator, List, Union

import rasterio.windows
from rasterio.windows import Window
from shapely.geometry import Polygon
import affine
import warnings
from pathlib import Path

import itertools
import numpy as np
import rasterio
import shapely
from shapely.geometry import Polygon
from PIL import Image as pilimg
from skimage import exposure, img_as_ubyte
from tqdm import tqdm


def get_chip_windows(raster_width: int,
                     raster_height: int,
                     raster_transform,
                     chip_width: int=256,
                     chip_height: int=256,
                     skip_partial_chips: bool=False,
                     ) -> Generator[Tuple[Window, affine.Affine, Polygon], any, None]:
    """Generator for rasterio windows of specified pixel size to iterate over an image in chips.
    Chips are created row wise, from top to bottom of the raster.
    Args:
        raster_width: rasterio meta['width']
        raster_height: rasterio meta['height']
        raster_transform: rasterio meta['transform']
        chip_width: Desired pixel width.
        chip_height: Desired pixel height.
        skip_partial_chips: Skip image chips at the edge of the raster that do not result in a full size chip.
    Returns :
        Yields tuple of rasterio chip window, chip transform and chip polygon.
    """
    col_row_offsets = itertools.product(range(0, raster_width, chip_width), range(0, raster_height, chip_height))
    raster_window = Window(col_off=0, row_off=0, width=raster_width, height=raster_height)

    for col_off, row_off in col_row_offsets:
        chip_window = Window(col_off=col_off, row_off=row_off, width=chip_width, height=chip_height)

        if skip_partial_chips:
            if row_off + chip_height > raster_height or col_off + chip_width > raster_width:
                continue

        chip_window = chip_window.intersection(raster_window)
        chip_transform = rasterio.windows.transform(chip_window, raster_transform)
        chip_bounds = rasterio.windows.bounds(chip_window, raster_transform)  # Uses transform of full raster.
        chip_poly = shapely.geometry.box(*chip_bounds, ccw=False)

        yield (chip_window, chip_transform, chip_poly)


def cut_chip_images(inpath_raster: Union[Path, str],
                    outpath_chipfolder: Union[Path, str],
                    chip_names: List[str],
                    chip_windows: List,
                    bands=[3, 2, 1]):
    """Cuts image raster to chips via the given windows and exports them to jpg."""

    src = rasterio.open(inpath_raster)

    all_chip_stats = {}
    for chip_name, chip_window in tqdm(zip(chip_names, chip_windows)):
        img_array = np.dstack(list(src.read(bands, window=chip_window)))
        img_array = exposure.rescale_intensity(img_array, in_range=(0, 2200))  # Sentinel2 range.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            img_array = img_as_ubyte(img_array)
        img_pil = pilimg.fromarray(img_array)

        # Export chip images
        Path(outpath_chipfolder).mkdir(parents=True, exist_ok=True)
        with open(Path(rf'{outpath_chipfolder}\{chip_name}.jpg'), 'w') as dst:
            img_pil.save(dst, format='JPEG', subsampling=0, quality=100)

        all_chip_stats[chip_name] = {'mean': img_array.mean(axis=(0, 1)),
                                     'std': img_array.std(axis=(0, 1))}
    src.close()

    return all_chip_stats


# In[ ]:






# In[3]:


import warnings
from typing import Union, Dict

import numpy as np
from geopandas import GeoDataFrame as GDF
from pandas import DataFrame as DF
import shapely
from shapely.geometry import Polygon
import rasterio.crs
import geopandas as gpd
from tqdm import tqdm


# In[4]:


def buffer_zero(ingeo: Union[GDF, Polygon]) -> Union[GDF, Polygon]:
    """Make invalid polygons (due to self-intersection) valid by buffering with 0."""
    if isinstance(ingeo, Polygon):
        if ingeo.is_valid is False:
            return ingeo.buffer(0)
        else:
            return ingeo
    elif isinstance(ingeo, GDF):
        if False in ingeo.geometry.is_valid.unique():
            ingeo.geometry = ingeo.geometry.apply(lambda _p: _p.buffer(0))
            return ingeo
        else:
            return ingeo


def close_holes(ingeo: Union[GDF, Polygon]) -> Union[GDF, Polygon]:
    """Close polygon holes by limitation to the exterior ring."""
    def _close_holes(poly: Polygon):
        if poly.interiors:
            return Polygon(list(poly.exterior.coords))
        else:
            return poly

    if isinstance(ingeo, Polygon):
        return _close_holes(ingeo)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _close_holes(_p))
        return ingeo


def set_crs(df: GDF, epsg_code: Union[int, str]) -> GDF:
    """Sets dataframe crs in geopandas pipeline.
    TODO: Deprecate with next rasterio version that will integrate set_crs method.
    """
    df.crs = {'init': f'epsg:{str(epsg_code)}'}
    return df


def explode_mp(df: GDF) -> GDF:
    """Explode all multi-polygon geometries in a geodataframe into individual polygon geometries.
    Adds exploded polygons as rows at the end of the geodataframe and resets its index.
    """
    outdf = df[df.geom_type == 'Polygon']

    df_mp = df[df.geom_type == 'MultiPolygon']
    for idx, row in df_mp.iterrows():
        df_temp = gpd.GeoDataFrame(columns=df_mp.columns)
        df_temp = df_temp.append([row] * len(row.geometry), ignore_index=True)
        for i in range(len(row.geometry)):
            df_temp.loc[i, 'geometry'] = row.geometry[i]
        outdf = outdf.append(df_temp, ignore_index=True)

    outdf.reset_index(drop=True, inplace=True)
    return outdf


def keep_biggest_poly(df: GDF) -> GDF:
    """Replaces MultiPolygons with the biggest polygon contained in the MultiPolygon."""
    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()
    for idx in row_idxs_mp:
        mp = df.loc[idx].geometry
        poly_areas = [p.area for p in mp]
        max_area_poly = mp[poly_areas.index(max(poly_areas))]
        df.loc[idx, 'geometry'] = max_area_poly
    return df


def clip(df: GDF,
         clip_poly: Polygon,
         explode_mp_: bool = False,
         keep_biggest_poly_: bool = False,
         ) -> GDF:
    """Filter and clip geodataframe to clipping geometry.
    The clipping geometry needs to be in the same projection as the geodataframe.
    Args:
        df: input geodataframe
        clip_poly: Clipping polygon geometry, needs to be in the same crs as the input geodataframe.
        explode_mp_: Applies explode_mp function. Append dataframe rows for each polygon in potential
            multipolygons that were created by the intersection. Resets the dataframe index!
        keep_biggest_poly_: Applies keep_biggest_poly function. Replaces MultiPolygons with the biggest
        polygon contained in the MultiPolygon.
    Returns:
        Result geodataframe.
    """
    df = df[df.geometry.intersects(clip_poly)].copy()
    df.geometry = df.geometry.apply(lambda _p: _p.intersection(clip_poly))
    # df = gpd.overlay(df, clip_poly, how='intersection')  # Slower.

    row_idxs_mp = df.index[df.geometry.geom_type == 'MultiPolygon'].tolist()

    if not row_idxs_mp:
        return df
    elif not explode_mp_ and (not keep_biggest_poly_):
        warnings.warn(f"Warning, intersection resulted in {len(row_idxs_mp)} split multipolygons. Use "
                      f"explode_mp_=True or keep_biggest_poly_=True.")
        return df
    elif explode_mp_ and keep_biggest_poly_:
        raise ValueError('You can only use one of "explode_mp_" or "keep_biggest_poly_"!')
    elif explode_mp_:
        return explode_mp(df)
    elif keep_biggest_poly_:
        return keep_biggest_poly(df)


def reclassify_col(df: Union[GDF, DF],
                   rcl_scheme: Dict,
                   col_classlabels: str= 'lcsub',
                   col_classids: str= 'lcsub_id',
                   drop_other_classes: bool=True
                   ) -> Union[GDF, DF]:
    """Reclassify class label and class ids in a dataframe column.
    # TODO: Simplify & make more efficient!
    Args:
        df: input geodataframe.
        rcl_scheme: Reclassification scheme, e.g. {'springcereal': [1,2,3], 'wintercereal': [10,11]}
        col_classlabels: column with class labels.
        col_classids: column with class ids.
        drop_other_classes: Drop classes that are not contained in the reclassification scheme.
    Returns:
        Result dataframe.
    """
    if drop_other_classes is True:
        classes_to_drop = [v for values in rcl_scheme.values() for v in values]
        df = df[df[col_classids].isin(classes_to_drop)].copy()

    rcl_dict = {}
    rcl_dict_id = {}
    for i, (key, value) in enumerate(rcl_scheme.items(), 1):
        for v in value:
            rcl_dict[v] = key
            rcl_dict_id[v] = i

    df[f'r_{col_classlabels}'] = df[col_classids].copy().map(rcl_dict)  # map name first, id second!
    df[f'r_{col_classids}'] = df[col_classids].map(rcl_dict_id)

    return df


reclass_legend = {
    'springcereal': [1, 2, 3, 4, 6, 7, 21, 55, 56, 210, 211, 212, 213, 214, 215, 224, 230, 234, 701, 702, 703, 704,
                     705],
    'wintercereal': [10, 11, 13, 14, 15, 16, 17, 22, 57, 220, 221, 222, 223, 235],
    'maize': [5, 216],
    'grassland': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 120, 121,
                  122, 123, 125, 126, 162, 170, 171, 172, 173, 174, 180, 182, 260, 261, 262, 263, 264, 266, 267,
                  268, 269, 270, 281, 282, 283, 284],
    'other': [23, 24, 25, 30, 31, 32, 35, 36, 40, 42, 51, 52, 53, 54, 55, 56, 57, 124, 160, 161, 280, 401, 402, 403,
              404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 415, 416, 417, 418, 420, 421, 422, 423, 424, 429,
              430, 431, 432, 434, 440, 448, 449, 450, 487, 488, 489, 491, 493, 496, 497, 498, 499, 501, 502, 503,
              504, 505, 507, 509, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
              528, 529, 530, 531, 532, 533, 534, 536, 539, 540, 541, 542, 543, 544, 545, 547, 548, 549, 550, 551,
              552, 553, 560, 561, 563, 570, 579]
    # drop other non-crop related classes (forest related, environment, recreation, other grass, permanent grass,
    # wasteland, ..)
    }


def reduce_precision(ingeo: Union[Polygon, GDF], precision: int=3) -> Union[Polygon, GDF]:
    """Reduces the number of after comma decimals of a shapely Polygon or geodataframe geometries.
    GeoJSON specification recommends 6 decimal places for latitude and longitude which equates to roughly 10cm of
    precision (https://github.com/perrygeo/geojson-precision).
    Args:
        ingeo: input geodataframe or shapely Polygon.
        precision: number of after comma values that should remain.
    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _reduce_precision(poly: Polygon, precision: int) -> Polygon:
        geojson = shapely.geometry.mapping(poly)
        geojson['coordinates'] = np.round(np.array(geojson['coordinates']), precision)
        poly = shapely.geometry.shape(geojson)
        if not poly.is_valid:  # Too low precision can potentially lead to invalid polygons due to line overlap effects.
            poly = poly.buffer(0)
        return poly

    if isinstance(ingeo, Polygon):
        return _reduce_precision(poly=ingeo, precision=precision)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _reduce_precision(poly=_p, precision=precision))
        return ingeo


def to_pixelcoords(ingeo: Union[Polygon, GDF],
                   reference_bounds: Union[rasterio.coords.BoundingBox, tuple],
                   scale: bool=False,
                   nrows: int=None,
                   ncols: int=None
                   ) -> Union[Polygon, GDF]:
    """Converts projected polygon coordinates to pixel coordinates of an image array.
    Subtracts point of origin, scales to pixelcoordinates.
    Input:
        ingeo: input geodataframe or shapely Polygon.
        reference_bounds:  Bounding box object or tuple of reference (e.g. image chip) in format (left, bottom,
            right, top)
        scale: Scale the polygons to the image size/resolution. Requires image array nrows and ncols parameters.
        nrows: image array nrows, required for scale.
        ncols: image array ncols, required for scale.
    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _to_pixelcoords(poly: Polygon, reference_bounds, scale, nrows, ncols):
        try:
            minx, miny, maxx, maxy = reference_bounds
            w_poly, h_poly = (maxx - minx, maxy - miny)
        except (TypeError, ValueError):
            raise Exception(
                f'reference_bounds argument is of type {type(reference_bounds)}, needs to be a tuple or rasterio bounding box '
                f'instance. Can be delineated from transform, nrows, ncols via rasterio.transform.reference_bounds')

        # Subtract point of origin of image bbox.
        x_coords, y_coords = poly.exterior.coords.xy
        p_origin = shapely.geometry.Polygon([[x - minx, y - miny] for x, y in zip(x_coords, y_coords)])

        if scale is False:
            return p_origin
        elif scale is True:
            if ncols is None or nrows is None:
                raise ValueError('ncols and nrows required for scale')
            x_scaler = ncols / w_poly
            y_scaler = nrows / h_poly
            return shapely.affinity.scale(p_origin, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))

    if isinstance(ingeo, Polygon):
        return _to_pixelcoords(poly=ingeo, reference_bounds=reference_bounds, scale=scale, nrows=nrows, ncols=ncols)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _to_pixelcoords(poly=_p, reference_bounds=reference_bounds,
                                                                         scale=scale, nrows=nrows, ncols=ncols))
        return ingeo


def invert_y_axis(ingeo: Union[Polygon, GDF],
                  reference_height: int
                  ) -> Union[Polygon, GDF]:
    """Invert y-axis of polygon or geodataframe geometries in reference to a bounding box e.g. of an image chip.
    Usage e.g. for COCOJson format.
    Args:
        ingeo: Input Polygon or geodataframe.
        reference_height: Height (in coordinates or rows) of reference object (polygon or image, e.g. image chip.
    Returns:
        Result polygon or geodataframe, same type as input.
    """
    def _invert_y_axis(poly: Polygon=ingeo, reference_height=reference_height):
        x_coords, y_coords = poly.exterior.coords.xy
        p_inverted_y_axis = shapely.geometry.Polygon([[x, reference_height - y] for x, y in zip(x_coords, y_coords)])
        return p_inverted_y_axis

    if isinstance(ingeo, Polygon):
        return _invert_y_axis(poly=ingeo, reference_height=reference_height)
    elif isinstance(ingeo, GDF):
        ingeo.geometry = ingeo.geometry.apply(lambda _p: _invert_y_axis(poly=_p, reference_height=reference_height))
        return ingeo


def cut_chip_geometries(vector_df, raster_width, raster_height, raster_transform, chip_width=128, chip_height=128, first_n_chips=None):
    """Workflow to cut a vector geodataframe to chip geometries.
    Filters small polygons and skips empty chips.
    Args:
        vector_df: Geodataframe containing the geometries to be cut to chip geometries.
        raster_width: rasterio meta['width']
        raster_height: rasterio meta['height']
        raster_transform: rasterio meta['transform']
        chip_width: Desired pixel width.
        chip_height: Desired pixel height.
        first_n_chips: Only processes the first n image chips, used for debugging.
    Returns: Dictionary containing the final chip_df, chip_window, chip_transform, chip_poly objects.
    """

    generator_window_bounds = utils.img.get_chip_windows(raster_width=raster_width,
                                                         raster_height=raster_height,
                                                         raster_transform=raster_transform,
                                                         chip_width=chip_width,
                                                         chip_height=chip_height,
                                                         skip_partial_chips=True)

    all_chip_dfs = {}
    for i, (chip_window, chip_transform, chip_poly) in enumerate(tqdm(generator_window_bounds)):
        if i >= first_n_chips:
            break

        # # Clip geometry to chip
        chip_df = vector_df.pipe(utils.geo.clip, clip_poly=chip_poly, keep_biggest_poly_=True)
        if not all(chip_df.geometry.is_empty):
            chip_df.geometry = chip_df.simplify(1, preserve_topology=True)
        else:
            continue
        # Drop small geometries
        chip_df = chip_df[chip_df.geometry.area * (10 * 10) > 5000]  #5000 sqm in UTM
        # Transform to chip pixelcoordinates and invert y-axis for COCO format.
        if not all(chip_df.geometry.is_empty):
            chip_df = chip_df.pipe(utils.geo.to_pixelcoords, reference_bounds=chip_poly.bounds, scale=True,
                                   ncols=chip_width, nrows=chip_height)
            chip_df = chip_df.pipe(invert_y_axis, reference_height=chip_height)
        else:
            continue

        chip_name = f'COCO_train2016_000000{100000+i}'  # _{clip_minX}_{clip_minY}_{clip_maxX}_{clip_maxY}'
        all_chip_dfs[chip_name] = {'chip_df': chip_df,
                                   'chip_window': chip_window,
                                   'chip_transform': chip_transform,
                                   'chip_poly': chip_poly}
    return all_chip_dfs


# In[ ]:






# In[ ]:


# %load farm-field/utils/other.py
#!/usr/bin/env python

# In[1]:


# other.py

from pathlib import Path
import pickle

import json


def new_pickle(outpath: Path, data):
    """(Over)write data to new pickle file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print(f'Writing new pickle file... {outpath.name}')


def load_pickle(inpath: Path):
    print(f'Loading from existing pickle file... {inpath.name}')
    with open(inpath, "rb") as f:
        return pickle.load(f)


def new_json(outpath: Path, data):
    """(Over)write data to new json file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=4)
    print(f'Writing new json file... {outpath.name}')


def load_json(inpath: Path):
    print(f'Loading from existing json file... {inpath.name}')
    with open(inpath, "r") as f:
        return json.load(f)


# In[ ]:






# In[ ]:


# %load farm-field/utils/coco.py
#!/usr/bin/env python

# In[ ]:


# %load farm-field/utils/other.py
#!/usr/bin/env python

# In[1]:


# other.py

from pathlib import Path
import pickle

import json


def new_pickle(outpath: Path, data):
    """(Over)write data to new pickle file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(data, f)
    print(f'Writing new pickle file... {outpath.name}')


def load_pickle(inpath: Path):
    print(f'Loading from existing pickle file... {inpath.name}')
    with open(inpath, "rb") as f:
        return pickle.load(f)


def new_json(outpath: Path, data):
    """(Over)write data to new json file."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(data, f, indent=4)
    print(f'Writing new json file... {outpath.name}')


def load_json(inpath: Path):
    print(f'Loading from existing json file... {inpath.name}')
    with open(inpath, "r") as f:
        return json.load(f)


# In[ ]:






# In[3]:


# coco.py

from typing import Union, Tuple, List, Dict
from pathlib import Path
import random
import itertools

from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from descartes import PolygonPatch
from PIL import Image as pilimage


# In[4]:


def train_test_split(chip_dfs: Dict, test_size=0.2, seed=1) -> Tuple[Dict, Dict]:
    """Split chips into training and test set.
    Args:
        chip_dfs: Dictionary containing key (filename of the chip) value (dataframe with
            geometries for that chip) pairs.
        test_size: Relative number of chips to be put in the test dataset. 1-test_size is the size of the
        training data set.
    """
    chips_list = list(chip_dfs.keys())
    random.seed(seed)
    random.shuffle(chips_list)
    split_idx = round(len(chips_list) * test_size)
    train_split = chips_list[split_idx:]
    val_split = chips_list[:split_idx]

    train_chip_dfs = {k: chip_dfs[k] for k in sorted(train_split)}
    val_chip_dfs = {k.replace('train', 'val'): chip_dfs[k] for k in sorted(val_split)}

    return train_chip_dfs, val_chip_dfs


def format_coco(chip_dfs: Dict, chip_width: int, chip_height: int):
    """Format train and test chip geometries to COCO json format.
    Args:
        chip_dfs: Dictionary containing key (filename of the chip) value (dataframe with
            geometries for that chip) pairs.
        chip_width: width of the chip in pixel size.
        chip_height: height of the chip in pixel size.
    COCOjson example structure and instructions below. For more detailed information on building a COCO
        dataset see http://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    cocojson = {
        "info": {...},
        "licenses": [...],
        "categories": [{"supercategory": "person","id": 1,"name": "person"},
                       {"supercategory": "vehicle","id": 2,"name": "bicycle"},
                       ...],
        "images":  [{"file_name": "000000289343.jpg", "height": 427, "width": 640, "id": 397133},
                    {"file_name": "000000037777.jpg", "height": 230, "width": 352, "id": 37777},
                    ...],
        "annotations": [{"segmentation": [[510.66,423.01,...,510.45,423.01]], "area": 702.10, "iscrowd": 0,
                         "image_id": 289343, "bbox": [473.07,395.93,38.65,28.67], "category_id": 18, "id": 1768},
                        {"segmentation": [[340.32,758.01,...,134.25,875.01]], "area": 342.08, "iscrowd": 0,
                         "image_id": 289343, "bbox": [473.07,395.93,38.65,28.67], "category_id": 18, "id": 1768},
                         ...]
        }
    - "id" in "categories" has to match "category_id" in "annotations".
    - "id" in "images" has to match "image_id" in "annotations".
    - "segmentation" in "annotations" is encoded in Run-Length-Encoding (except for crowd region (iscrowd=1)).
    - "id" in "annotations has to be unique for each geometry, so 4370 geometries in 1000 chips > 4370 unique
       geometry ids. However, does not have to be unique between coco train and validation set.
    - "file_name" in "images" does officially not have to match the "image_id" in "annotations" but is strongly
       recommended.
    """
    cocojson = {
        "info": {},
        "licenses": [],
        'categories': [{'supercategory': 'AgriculturalFields',
                        'id': 1,  # needs to match category_id.
                        'name': 'agfields_singleclass'}]}

    annotation_id = 1

    for chip_name in chip_dfs.keys():

        if 'train' in chip_name:
            chip_id = int(chip_name[21:])
        elif 'val' in chip_name:
            chip_id = int(chip_name[19:])

        image = {"file_name": f'{chip_name}.jpg',
                  "id": int(chip_id),
                  "height": chip_width,
                  "width": chip_height}
        cocojson.setdefault('images', []).append(image)

        for _, row in chip_dfs[chip_name]['chip_df'].iterrows():
            # Convert geometry to COCO segmentation format:
            # From shapely POLYGON ((x y, x1 y2, ..)) to COCO [[x, y, x1, y1, ..]].
            # The annotations were encoded by RLE, except for crowd region (iscrowd=1)
            coco_xy = list(itertools.chain.from_iterable((x, y) for x, y in zip(*row.geometry.exterior.coords.xy)))
            coco_xy = [round(coords, 2) for coords in coco_xy]
            # Add COCO bbox in format [minx, miny, width, height]
            bounds = row.geometry.bounds  # COCO bbox
            coco_bbox = [bounds[0], bounds[1], bounds[2] - bounds[0], bounds[3] - bounds[1]]
            coco_bbox = [round(coords, 2) for coords in coco_bbox]

            annotation = {"id": annotation_id,
                           "image_id": int(chip_id),
                           "category_id": 1,  # with multiple classes use "category_id" : row.reclass_id
                           "mycategory_name": 'agfields_singleclass',
                           "old_multiclass_category_name": row['r_lc_name'],
                           "old_multiclass_category_id": row['r_lc_id'],
                           "bbox": coco_bbox,
                           "area": row.geometry.area,
                           "iscrowd": 0,
                           "segmentation": [coco_xy]}
            cocojson.setdefault('annotations', []).append(annotation)

            annotation_id += 1

    return cocojson


def move_coco_val_images(inpath_train_folder, val_chips_list):
    """Move validation chip images to val folder (applies train/val split on images)
    Args:
        inpath_train_folder: Filepath to the training COCO image chip "train" folder
        val_chips_list: List of validation image key names that should be moved.
    """
    outpath_val_folder = inpath_train_folder.parent / 'val2016'
    Path(outpath_val_folder).mkdir(parents=True, exist_ok=True)
    for chip in val_chips_list:
        Path(rf'{inpath_train_folder}\{chip.replace("val", "train")}.jpg').replace(rf'{outpath_val_folder}\{chip}.jpg')


def coco_to_shapely(inpath_json: Union[Path, str],
                    categories: List[int] = None) -> Dict:
    """Transforms COCO annotations to shapely geometry format.
    Args:
        inpath_json: Input filepath coco json file.
        categories: Categories will filter to specific categories and images that contain at least one
        annotation of that category.
    Returns:
        Dictionary of image key and shapely Multipolygon.
    """

    data = utils.other.load_json(inpath_json)
    if categories is not None:
        # Get image ids/file names that contain at least one annotation of the selected categories.
        image_ids = sorted(list(set([x['image_id'] for x in data['annotations'] if x['category_id'] in categories])))
    else:
        image_ids = sorted(list(set([x['image_id'] for x in data['annotations']])))
    file_names = [x['file_name'] for x in data['images'] if x['id'] in image_ids]

    # Extract selected annotations per image.
    extracted_geometries = {}
    for image_id, file_name in zip(image_ids, file_names):
        annotations = [x for x in data['annotations'] if x['image_id'] == image_id]
        if categories is not None:
            annotations = [x for x in annotations if x['category_id'] in categories]

        segments = [segment['segmentation'][0] for segment in annotations]  # format [x,y,x1,y1,...]

        # Create shapely Multipolygons from COCO format polygons.
        mp = MultiPolygon([Polygon(np.array(segment).reshape((int(len(segment) / 2), 2))) for segment in segments])
        extracted_geometries[str(file_name)] = mp

    return extracted_geometries


def plot_coco(inpath_json, inpath_image_folder, start=0, end=2):
    """Plot COCO annotations and image chips"""
    extracted = utils.coco.coco_to_shapely(inpath_json)

    for key in sorted(extracted.keys())[start:end]:
        print(key)
        plt.figure(figsize=(5, 5))
        plt.axis('off')

        img = np.asarray(pilimage.open(rf'{inpath_image_folder}\{key}'))
        plt.imshow(img, interpolation='none')

        mp = extracted[key]
        patches = [PolygonPatch(p, ec='r', fill=False, alpha=1, lw=0.7, zorder=1) for p in mp]
        plt.gca().add_collection(PatchCollection(patches, match_original=True))
        plt.show()


# In[ ]:






# In[ ]:




