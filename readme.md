We have already made the texture maps available on our [database portal](https://opensvbrdf.github.io/). Here, we attach the calculation method of the GGX anisotropic BRDF used in our entire system, for everyone to use as a reference when rendering data.


First, a local coordinate system (onb) for rendering is established based on the normal and tangent. Then, $\omega_{i}$, $\omega_{o}$, onb, and other parameters are used as inputs to call the GGX_BRDF.eval() method, which will return the BRDF value. 

Our implementation references the formulas in the article [Physically-based shading at Disney](https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf).


The rest of the code is currently being organized and will be coming soon. If you’re interested in our project, we warmly welcome you to star our repository. This will make it easier for you to find our updates in the future.