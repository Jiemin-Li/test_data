import numpy as np
import scipy
from matplotlib import pyplot as plt


class SpectralImageGenerator():
    """A class that will be used to generate a set of 'spectral images'.

    This class is used to hold a series of 'mask layers' and apply these to a
    set of 'spectra' to produce 'spectral images' where different regions on
    the sample have a mixture of different spectra. This can be used for
    generating testing data for the data analysis/ processing routines for the
    ARI beamline at NSLS-II.

    Parameters
    ----------
    *args : list
        List of args used to initialize the class, see the docstring for
        self.__init__() for details.
    *kwargs :list
        List of kwargs used to initialize the class, see the docstring for
        self.__init__() for details.

    Attributes:
    num_layers : int
        The number of mask/spectra 'layers' (or elements) in the mask_layers
        list.
    shape : (n, m)
        Two element Tuple describing the height (n) and width (m) array
        dimensions of the output mask layers, masks, spectral layers and
        spectral images. This is the equivalent of what is returned by
        numpy.array.shape(), but limited to a 2D numpy array.
    masks : list
        List of `num_layers` 2D 'mask layers' (n x m numpy.arrays) generated
        on instantiation. Each 'mask layer' will be a 2D image with the
        value at each point being between 0 and 1 the sum of all layers will
        be a 2D image with every point approximatly equal to 1 and they are
        normalized versions of raw_masks with the final layer being a unitary
        image - sum of all others.
    raw_masks : list
        List of `num_layers`-1 2D 'mask layers' (n x m numpy.arrays) generated
        on instantiation. Each 'mask layer' will be a 2D image with the
        value at each point being between 0 and 1 the sum of all layers will
        be a 2D image with every point approximatly equal to 1.

    Methods:
    visualize_masks(cmap=['Reds', 'Blues', 'Greens', 'Purples']):
        A method that displays a plot of the self.masks and self.raw_masks
        data for visualization reasons.
    """

    def __init__(self, num_layers=4, shape=(25, 25), seed=4,
                 region_size=1):
        """Initialization function for the SpectralImageGenerator object

        A method called at instantiation that generates 'values' for the
        attributes: masks, raw_masks, spectra, .....

        Parameters
        ----------
        num_layers : int, optional.
            The number of mask/spectra 'layers' to be generated using the
            layer image generation function. Default value is 4.
        shape : (n, m), optional.
            Two element Tuple describing the height (n) and width (m) array
            dimensions of the output mask layers, masks, spectral layers and
            spectral images. This is the equivalent of what is returned by
            numpy.array.shape(), but limited to a 2D numpy array. Default
            value is (10, 10).
        seed : int, optional, default = 4
            Initializes numpy's random number generator to the specified state.
            For the default data set the seed is 4, a seed of `None` will
            generate a random set of layers, any other number will provide a
            specific random set of layers.
        region_size :
            Controls the image morphology.  A higher number results in
            a larger number of small regions.  If a list is supplied then the
            regions are anisotropic.
        """

        self.num_layers = num_layers
        self.shape = shape

        _masks, _raw_masks = self._generate_masks(shape=shape,
                                                  region_size=region_size,
                                                  num_layers=num_layers,
                                                  seed=seed)
        self.masks = _masks
        self.raw_masks = _raw_masks

    def _organic_image(self, shape, region_size=1):
        """
        Generates an image containing amorphous blobs

        Parameters
        ----------
        shape : list
            The size of the image to generate in [Nx, Ny] where N is the
            number of pixels
        region_size :
            Controls the image morphology.  A higher number results in
            a larger number of small regions.  If a list is supplied then the
            regions are anisotropic.

        Returns
        -------
        image : ndarray
            A boolean array with ``True`` values denoting the pore space

        See Also
        --------
        make_uniform

        Notes
        -----
        This function generates random noise, the applies a gaussian blur to
        the noise with a sigma controlled by the region_size argument as:

            $$ np.mean(shape) / (8 * region_size) $$

        The value of 8 was chosen so that a ``region_size`` of 1 gave a
        reasonable result for the default 200x200 grid.
        """

        def make_uniform(image, scale=[0, 1]):
            """
            Converts a grey-scale image to a uniform, flat, distribution.

            Parameters
            ----------
            image : ndarray
                Greyscale image to be flattened.
            scale : [low, high]
                A list indicating the lower and upper bounds randomly
                distributed output data. The default is [0, 1].

            Returns
            -------
            output : ndarray
                A uniformly distributed copy of image spanning the values
                from 'scale'.
            """
            argsort_image = np.argsort(np.argsort(image.flatten()))
            linspace_image = np.linspace(scale[0], scale[1], len(argsort_image),
                                         endpoint=True)
            uniform_flatten_image = linspace_image[argsort_image]
            image = np.reshape(uniform_flatten_image, image.shape)
            return image

        if isinstance(shape, int):
            shape = [shape] * 3
        if len(shape) == 1:
            shape = [shape[0]] * 3
        shape = np.array(shape)
        if isinstance(region_size, int):
            region_size = [region_size] * len(shape)
        region_size = np.array(region_size)
        sigma = np.mean(shape) / (8 * region_size)
        image = np.random.random(shape)
        image = scipy.ndimage.gaussian_filter(image, sigma=sigma)
        image = make_uniform(image)

        return image

    def _generate_masks(self, shape=(25, 25), region_size=1, num_layers=4,
                        seed=4):
        """
        Returns a set of masks for simulated spectroscopic imaging datasets

        This function returns a set of `num_regions` layers that can be used as
        multiplicative masks to distribute different 'spectra' accross a defined
        'image' to generate simulated spectroscopic imaging datasets. Each
        'mask' is an 'n x m' image with each point value being between 0 and 1.
        If each mask is 'summed' together the result is an image with each point
        being 1, to ensure that the resulting spectroscopic dataset has spectra
        at each point in the image.

        Each mask (except the last) is randomnly generated from `generate_mask`,
        and stored in the returned `raw_masks` list. It then has all previous
        masks subtracted from it (with 0 as floor) to ensure that there are a
        reasonable mixing of each of the 'regions'. All masks (except the last)
        are then normalized to the maximum value of the sum of all masks to
        ensure that the sum of all masks has values between 0 and 1. A final
        mask is then generated by subtracting the sum of all masks from a numpy
        array, with shape=shape, and all values equa1 to 1. This last step
        ensures that the sum of all masks has a value of 1 everywhere. These
        masks are returned in the `masks` list.

        Parameters
        ----------
        shape : (int, int), optional.
            A 2 element tuple giving the 'n' (vertical) and 'm' (horizontal)
            number of pixels for the generated 'masks'. Default value is (200,
            200).
        region_size : float, optional.
            Passed to `self._organic_image`, which is used to generate each
            'mask' layer. see `generate_mask` docstring for explanation. Default
            value is 0.2, works best for 0< region_size< 1.
        num_layers : int, optional.
            Passed to `self._organic_image`, which is used to generate each
            'mask' layer. see `self._organic_image` docstring for explanation.
            Default value is 4.
        seed : int, optional, default = 4
            Initializes numpy's random number generator to the specified state.
            For the default data set the seed is 4, a seed of `None` will
            generate a random set of layers, any other number will provide a
            specific random set of layers.


        Returns
        -------
        masks: [numpy.array, ..., numpy.array].
            A list of `num_regions` normalized masks randomnly generated using
            `generate_mask` with each (except the final mask) having all
            previous masks subtracted from it. The final mask is then found by
            subtracting all previous masks from a numpy.ones(shape) array.
        raw_masks: [numpy.array, ..., numpy.array].
            A list of `num_regions-1` masks randomnly generated using
            `generate_mask`. These are used as the basis for the masks found in
            'masks'
        """

        np.random.seed(seed)

        summed_mask = np.zeros(shape)
        raw_masks = []
        masks = []

        for i in range(num_layers - 1):
            raw_masks.append(self._organic_image(shape,
                                                 region_size=region_size))
            if i == 0:
                masks.append(raw_masks[i])
            else:
                masks.append(np.array([[max(0, a - b)
                                        for a, b in zip(a_row, b_row)]
                                       for a_row, b_row in zip(raw_masks[i],
                                                               summed_mask)]))
            summed_mask += raw_masks[i]

        for i in range(len(masks)):
            masks[i] /= summed_mask.max()

        summed_mask /= summed_mask.max()

        final_mask = np.array([[max(0, a - b) for a, b in zip(a_row, b_row)]
                               for a_row, b_row in zip(np.ones(shape),
                                                       summed_mask)])
        masks.append(final_mask)

        summed_mask += final_mask

        normalize_mask = np.zeros(shape)
        for i in range(len(masks)):
            normalize_mask += masks[i]

        for i in range(len(masks)):
            masks[i] /= normalize_mask
            np.around(masks[i], decimals=4)

        return masks, raw_masks

    def visualize_masks(self, cmaps=['Reds', 'Blues', 'Greens', 'Purples']):
        """
        Used to visualize the output of generate_masks

        This function takes in the returned values from generate_masks and plots
        them as different colour heat-maps. The first row of figures are the
        `raw_masks`, the second row are the individual `masks` and the final row
        is all of the `masks` on the same axes.

        Parameters
        ----------
        cmaps: [str, ..., str].
            A list of strings corresponding to matplotlib colour maps to be used
            for each layer, the length of this list must be greateer than the
            length of the 'mask' list. The default works for up to 4 layers,
            other colour map options can be found at this link:

            https://matplotlib.org/stable/users/explain/colors/colormaps.html


        Returns
        -------
        figures: [matplotlib.figure.Figure, matplotlib.figure.Figure,
                  matplotlib.figure.Figure].
            A list of matplotlib.figure.Figure objects for each row of the
            outputted plots.
        """

        if len(self.masks) > len(cmaps):
            raise ValueError(f'The number of colour maps in `cmaps` (currently '
                             f'{len(cmaps)}) needs to be greater than the '
                             f'number of masks in `masks` (currently '
                             f'{len(self.masks)})')

        figure, axes = plt.subplots(3, len(self.masks),
                                    figsize=[len(self.masks) * 2.5, 10],
                                    layout='tight', sharey=True)

        figure.text(0, 0.967,
                    'Raw randomnly generated masks for the `n-1` masks',
                    fontsize=16)
        axes[0, 0].set_ylabel(r'sample y ($\mu$m)')
        for i in range(len(self.raw_masks)):
            axes[0, i].set_xlabel(r'sample x ($\mu$m)')
            axes[0, i].set_title(f'Layer {i + 1}')
        axes[0, -1].set_axis_off()

        figure.text(0, 0.645,
                    'Normalized generated masks for each of the `n` masks',
                    fontsize=16)
        axes[1, 0].set_ylabel(r'sample y ($\mu$m)')
        for i in range(len(self.masks)):
            axes[1, i].set_xlabel(r'sample x ($\mu$m)')
            axes[1, i].set_title(f'Layer {i + 1}')

        figure.text(0, 0.3, 'Combined generated masks', fontsize=16)
        axes[2, 0].set_ylabel(r'sample y ($\mu$m)')
        axes[2, 0].set_xlabel(r'sample x ($\mu$m)')
        for i in range(1, len(self.masks)):
            axes[2, i].set_axis_off()

        for i, raw_mask in enumerate(self.raw_masks):
            axes[0][i].imshow(raw_mask, origin='lower', interpolation='none',
                              cmap=cmaps[i],
                              extent=[-raw_mask.shape[1] / 20,
                                      raw_mask.shape[1] / 20,
                                      -raw_mask.shape[0] / 20,
                                      raw_mask.shape[0] / 20])

        for i, mask in enumerate(self.masks):
            axes[1, i].imshow(mask, origin='lower', interpolation='none',
                              cmap=cmaps[i],
                              extent=[-mask.shape[1] / 20, mask.shape[1] / 20,
                                      -mask.shape[0] / 20, mask.shape[0] / 20])
            axes[2, 0].imshow(mask, origin='lower', alpha=(mask / mask.max()),
                              cmap=cmaps[i],
                              extent=[-raw_mask.shape[1] / 20,
                                      raw_mask.shape[1] / 20,
                                      -raw_mask.shape[0] / 20,
                                      raw_mask.shape[0] / 20])

        return figure
