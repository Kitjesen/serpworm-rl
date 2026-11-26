// Local optional BeamAdapter extension. Stock libraries are never overwritten.
// BeamAdapter uses the same J for torsional stiffness and rotational inertia.
// Preserve its polar J for inertia; compensate G in the stiffness call so that
// G_effective * J_polar = G_physical * J_SaintVenant. Current rectangular
// BeamAdapter sections have Asy=Asz=0 (no transverse shear term). This adapter
// must be revised if that implementation changes; it is not a physical Poisson
// ratio change. Public defaultPoissonRatio remains the material's real value.
#include <BeamAdapter/component/BeamInterpolation.h>
#include <sofa/core/ObjectFactory.h>
#include <cmath>
#include <algorithm>

class RectangularStripInterpolation final : public beamadapter::BeamInterpolation<sofa::defaulttype::Rigid3Types>
{
public:
    using Parent = beamadapter::BeamInterpolation<sofa::defaulttype::Rigid3Types>;
    SOFA_CLASS(RectangularStripInterpolation, Parent);
    void getMechanicalParameters(sofa::Index i, Real& young, Real& poisson, Real& density) override
    {
        Parent::getMechanicalParameters(i, young, poisson, density);
        const auto& section = this->getBeamSection(i);
        if (this->crossSectionShape.getValue().getSelectedItem() != "rectangular"
            || section._Asy != 0 || section._Asz != 0)
            throw std::runtime_error("StripSection requires rectangular BeamAdapter with zero shear areas");
        const double y = std::sqrt(12*section._Iz/section._A);
        const double z = std::sqrt(12*section._Iy/section._A);
        const double w = std::max(y,z), h = std::min(y,z), r = h/w;
        const double jsv = w*h*h*h*(1.0/3.0-0.21*r*(1-std::pow(r,4)/12));
        poisson = (1+poisson)*section._J/jsv-1;
    }
};

extern "C" {
void initExternalModule() {}
const char* getModuleName() { return "StripSection"; }
const char* getModuleVersion() { return "0.1"; }
const char* getModuleLicense() { return "LGPL"; }
const char* getModuleDescription() { return "Rectangular strip torsion correction preserving polar inertia"; }
void registerObjects(sofa::core::ObjectFactory* factory)
{
    factory->registerObjects(sofa::core::ObjectRegistrationData("Saint-Venant torsion for thin rectangular strips")
        .add<RectangularStripInterpolation>());
}
}
