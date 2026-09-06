"""Independent feature-evidence families.

Target P0.2: migrate SIFT + RootSIFT primitives. P1 adds ORB and AKAZE as
separate evidence families (independent implementation paths with different
descriptor failure modes). Ridge/texture structure detectors from the source
are deferred and re-framed as complementary gradient/line evidence, not lunar
terrain landmarks.
"""