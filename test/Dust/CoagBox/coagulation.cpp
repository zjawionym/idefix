#include "idefix.hpp"
#include "coagulation.hpp"
#include "npy.hpp"

Coagulation::Coagulation(Input &input) {
  const int  kpol = input.GetOrSet<int>("Coala","kpol",0,0);
  if(kpol!=0) {
    IDEFIX_ERROR("Only kpol = 0 is available when using Coala in Idefix");
  }
  const int  nSpecies = input.Get<int>("Dust","nSpecies",0);
  const real DENSITY  = input.GetOrSet<real>("Units","density",0,1.);
  const real LENGTH   = input.GetOrSet<real>("Units","length",0,1.);
  this->kernel   = input.GetOrSet<int>("Coala","kernel",0,3);
  this->eps      = input.GetOrSet<real>("Coala","eps",0,1e-20)/DENSITY*LENGTH*LENGTH*LENGTH;
  this->coagCFL  = input.GetOrSet<real>("Coala","coagCFL",0,0.5);

  idfx::cout << "Have dust coagulation with nSpecies = " << nSpecies << " dust species" << std::endl;
  idfx::cout << "    with : kernel = " << kernel << " --- eps = " << eps << " --- coagCFL = " << coagCFL << std::endl;

  //==============================================//
  //==============================================//
  // In this part, we load the coagulation tables //
  //==============================================//
  //==============================================//

  //===== Load the mass grid =====// 
  std::string filename_massgrid = "data/test-massgrid-kernel-"+std::to_string(kernel)+".npy";
  std::vector<uint64_t> shape_massgrid;
  bool fortran_order_massgrid;
  std::vector<double> massgrid;
  npy::LoadArrayFromNumpy(filename_massgrid, shape_massgrid, fortran_order_massgrid, massgrid);
    //===== verification =====//
  if(massgrid.size() != static_cast<size_t>(nSpecies+1)) {
    IDEFIX_ERROR("Mass grid has an unexpected size");
  }
    //========================//

  this->binWidth = IdefixArray1D<real>("binWidth", nSpecies);
  IdefixHostArray1D<real> binWidthHost = Kokkos::create_mirror_view(this->binWidth);
  for(int n = 0; n <nSpecies; n++) {
    binWidthHost(n) = massgrid[n+1] - massgrid[n];
  }
  Kokkos::deep_copy(this->binWidth, binWidthHost);

  //===== Loading the Flux Table =====//
  std::string filename_coagTabFlux = "data/test-coagTabFlux-kernel-"+std::to_string(kernel)+".npy";
  std::vector<uint64_t> shape_coagTabFlux;
  bool fortran_order_coagTabFlux;
  std::vector<double> coagTabFlux;
  npy::LoadArrayFromNumpy(filename_coagTabFlux, shape_coagTabFlux, fortran_order_coagTabFlux, coagTabFlux);
    //===== verification =====//
  const size_t expectedSize = static_cast<size_t>(nSpecies)*static_cast<size_t>(nSpecies)*static_cast<size_t>(nSpecies);
  if(coagTabFlux.size() != expectedSize) {
    IDEFIX_ERROR("Coagulation flux table has an unexpected size");
  }
    //========================//

  this->coagTabFlux = IdefixArray3D<real>("coagTabFlux", nSpecies, nSpecies, nSpecies);
  IdefixHostArray3D<real> coagTabFluxHost = Kokkos::create_mirror_view(this->coagTabFlux);
  for(int i=0; i<nSpecies; i++) {
    for(int j=0; j<nSpecies; j++) {
      for(int k=0; k<nSpecies; k++) {
        int index = i*nSpecies*nSpecies + j*nSpecies + k;
        coagTabFluxHost(i,j,k) = coagTabFlux[index];
      }
    }
  }
  Kokkos::deep_copy(this->coagTabFlux, coagTabFluxHost);
}


void Coagulation::AddSourceTerm(DataBlock *data, const real t, const real dtin) {
  //==============================================//
  
  // Coagulation source term : compute the coagulation product to finally upload the dust densities. 
  // This function is implemented via operator splitting : coagulation is integrated in a step.
  // Hence, enroll the function with `data.EnrollUserStepFirst(&CoagSourceTerm);`.
  // This version does not preserve bins momentum yet.
  
  //==============================================//
  data->PrimToCons();
  IdefixArray1D<real> h = this->binWidth;
  IdefixArray3D<real> coagTabFlux = this->coagTabFlux;
  const int nSpecies = data->dust.size();
  auto AllUc = data->dust[0]->AllUc;

  const real eps = this->eps;
  const real coagCFL = this->coagCFL;
  const bool physicalKernel = (this->kernel==3);


  typedef Kokkos::TeamPolicy<>              team_policy;
  typedef Kokkos::TeamPolicy<>::member_type member_type;
  typedef Kokkos::DefaultExecutionSpace::scratch_memory_space ScratchSpace;
  typedef Kokkos::View<real*, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchViewType1D;
  typedef Kokkos::View<real**, ScratchSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> ScratchViewType2D;
  int scratch_size = 5*ScratchViewType1D::shmem_size(nSpecies*sizeof(real))
                     + ScratchViewType2D::shmem_size(nSpecies*sizeof(real), nSpecies*sizeof(real))
                     + ScratchViewType1D::shmem_size((nSpecies+1)*sizeof(real)); 
  
  const int NK = data->np_tot[KDIR];
  const int NJ = data->np_tot[JDIR];
  const int NI = data->np_tot[IDIR];
  const int NKNJNI = NK*NJ*NI;
  const int NJNI = NJ*NI;

  const int level = 1;
  Kokkos::parallel_for(
    "Coagulation::AddSourceTerm_Space",
    team_policy(NKNJNI, Kokkos::AUTO).set_scratch_size(level, Kokkos::PerTeam(scratch_size)),
    KOKKOS_LAMBDA (member_type team_member) {
      int k = team_member.league_rank()/NJNI;
      int j = (team_member.league_rank() - k*NJNI)/NI;
      int i = team_member.league_rank() - k*NJNI - j*NI;

      ScratchViewType1D rho(team_member.team_scratch(level), nSpecies);
      ScratchViewType1D g0(team_member.team_scratch(level), nSpecies);
      ScratchViewType1D vx1(team_member.team_scratch(level), nSpecies);
      ScratchViewType1D vx2(team_member.team_scratch(level), nSpecies);
      ScratchViewType1D vx3(team_member.team_scratch(level), nSpecies);
      ScratchViewType2D deltaV(team_member.team_scratch(level), nSpecies, nSpecies);
      ScratchViewType1D Flux(team_member.team_scratch(level), nSpecies+1);

      // Initial State
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, 0, nSpecies),
        [&] (int n) {
          rho(n) = AllUc(n,RHO,k,j,i);
          vx1(n) = AllUc(n,MX1,k,j,i)/rho(n);
          vx2(n) = AllUc(n,MX2,k,j,i)/rho(n);
          vx3(n) = AllUc(n,MX3,k,j,i)/rho(n);
        }
      );
      team_member.team_barrier();

      // Relative velocity
      if(physicalKernel) {
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, 0, nSpecies),
          [&] (int lp) {
            for(int l=0; l<nSpecies; l++) {
              real dvx = vx1(lp) - vx1(l);
              real dvy = vx2(lp) - vx2(l);
              real dvz = vx3(lp) - vx3(l);
              deltaV(lp,l) = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
            }
          }
        );
      }
      team_member.team_barrier();

      //===== COAGULATION SUBCYCLING =====//
      real time = 0.0;
      int count = 0;
      while(time<dtin) {
        Flux(0) = 0.0;     // No mass flux crossing the left boundary of the distribution
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, 0, nSpecies),
          [&] (int n) {
            g0(n) = rho(n)/h(n);
          }
        );
        team_member.team_barrier();

        //===== FLUX CALCULATION =====//
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, 0, nSpecies),
          [&] (int n) {
            real flux = 0.0;
            for(int lp=0; lp <= n; lp++) {
              real g0_lp = g0(lp);
              for(int l=0; l<nSpecies; l++) {
                real g0_l = g0(l);
                real delta_v = physicalKernel ? deltaV(lp,l) : 1.0;
                flux += g0_lp*g0_l*coagTabFlux(n,lp,l)*delta_v;
              }
            }
            Flux(n+1) = flux;
          }
        );
        team_member.team_barrier();

        //===== CFL CALCULATION =====//
        real dt_sub = Kokkos::Experimental::finite_max<real>::value;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team_member, 0, nSpecies),
          [&] (int n, real &local_min) {
            real dflux = Flux(n+1) - Flux(n);
            if(Kokkos::fabs(dflux)>0.0 && rho(n)>eps*h(n)) {
              local_min = Kokkos::fmin(Kokkos::fabs(rho(n)/dflux), local_min);
            }
          }
        , Kokkos::Min<real>(dt_sub));
        real dt_coag = dt_sub*coagCFL;
        dt_coag = Kokkos::fmin(dt_coag, dtin-time);     // Last substep
        if(dt_coag<=0.0) {
          Kokkos::abort("Error: coagulation time step is non-positive");
        }

        //===== EULER UPDATE =====//
        Kokkos::parallel_for(
          Kokkos::TeamThreadRange(team_member, 0, nSpecies),
          [&] (int n) {
          rho(n) = Kokkos::fmax(rho(n) - dt_coag*(Flux(n+1) - Flux(n)), eps*h(n));
          }
        );
        team_member.team_barrier();
        time += dt_coag;
        count += 1;
      }

      //===== UPDATE THE FINAL DENSITIES =====//
      Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team_member, 0, nSpecies),
        [&] (int n) {
        AllUc(n,RHO,k,j,i) = rho(n);
        // To keep a constant velocity (just for the test with kernel = 3)
        // If we don't do that, the update on densities leads to update on velocities
        // We thus update the momentum with new densities in order to keep velocities (and dv) constants
        AllUc(n,MX1,k,j,i) = rho(n)*vx1(n);
        AllUc(n,MX2,k,j,i) = rho(n)*vx2(n);
        AllUc(n,MX3,k,j,i) = rho(n)*vx3(n);
        }
      );
    }
  );
  data->ConsToPrim();
}

/*
void Coagulation::AddSourceTerm(DataBlock *data, const real t, const real dtin) {
  //==============================================//
  
  // Coagulation source term : compute the coagulation product to finally upload the dust densities. 
  // This function is implemented via operator splitting : coagulation is integrated in a step.
  // Hence, enroll the function with `data.EnrollUserStepFirst(&CoagSourceTerm);`.
  // This version does not preserve bins momentum yet.
  // This version does not work on GPU yet.
  
  //==============================================//
  data->PrimToCons();
  IdefixArray1D<real> h = this->binWidth;
  IdefixArray3D<real> coagTabFlux = this->coagTabFlux;
  const int nSpecies = data->dust.size();
  auto AllUc = data->dust[0]->AllUc;
  const int test_size = 20;  // Because we can't give nSpecies in the KOKKOS_LAMBDA

  const real eps = this->eps;
  const real coagCFL = this->coagCFL;
  const bool physicalKernel = (this->kernel==3);

  idefix_for("Coagulation::AddSourceTerm",0,data->np_tot[KDIR],0,data->np_tot[JDIR],0,data->np_tot[IDIR],
            KOKKOS_LAMBDA (int k, int j, int i) {
              real rho[test_size];
              real g0[test_size];
              real vx1[test_size];
              real vx2[test_size];
              real vx3[test_size];
              real deltaV[test_size][test_size];
              real Flux[test_size+1];

              // Initial State
              for(int n=0; n<nSpecies; n++) {
                rho[n] = AllUc(n,RHO,k,j,i);
                vx1[n] = AllUc(n,MX1,k,j,i)/rho[n];
                vx2[n] = AllUc(n,MX2,k,j,i)/rho[n];
                vx3[n] = AllUc(n,MX3,k,j,i)/rho[n];
              }
              

              // Relative velocity
              if(physicalKernel) {
                for(int lp=0; lp<nSpecies; lp++) {
                  for(int l=0; l<nSpecies; l++) {
                    real dvx = vx1[lp] - vx1[l];
                    real dvy = vx2[lp] - vx2[l];
                    real dvz = vx3[lp] - vx3[l];
                    deltaV[lp][l] = sqrt(dvx*dvx + dvy*dvy + dvz*dvz);
                  }
                }
              }

              //===== COAGULATION SUBCYCLING =====//
              real time = 0.0;
              while(time<dtin) {
                Flux[0] = 0.0;  // No mass flux crossing the left boundary of the distribution
                for(int n=0; n<nSpecies; n++) {
                  g0[n] = rho[n]/h(n);
                }
                //===== FLUX CALCULATION =====//
                for(int n=0; n<nSpecies; n++) {
                  real flux = 0.0;
                  for(int lp=0; lp <= n; lp++) {
                    real g0_lp = g0[lp];
                    for(int l=0; l<nSpecies; l++) {
                      real g0_l = g0[l];
                      real delta_v = physicalKernel ? deltaV[lp][l] : 1.0;
                      flux += g0_lp*g0_l*coagTabFlux(n,lp,l)*delta_v;
                    }
                  }
                  Flux[n+1] = flux;
                }
                //===== CFL CALCULATION =====//
                real dt_sub = std::numeric_limits<double>::max();
                for(int n=0; n<nSpecies; n++) {
                  real dflux = Flux[n+1] - Flux[n];
                  if(std::abs(dflux)>0.0 && rho[n]>eps*h(n)) {
                    dt_sub = std::min(dt_sub, std::abs(rho[n]/dflux));
                  }
                }
                real dt_coag = dt_sub*coagCFL;
                dt_coag = std::min(dt_coag, dtin-time);  // Last substep
                if(dt_coag<=0.0) {
                  IDEFIX_ERROR("Coagulation time step is non-positive");
                }
                //===== EULER UPDATE =====//
                for(int n=0; n<nSpecies; n++) {
                  rho[n] = std::max(rho[n] - dt_coag*(Flux[n+1] - Flux[n]), eps*h(n));
                }
                time += dt_coag;
              }
              //===== UPDATE THE FINAL DENSITIES =====//
              for(int n=0; n<nSpecies; n++) {
                AllUc(n,RHO,k,j,i) = rho[n];
                // To keep a constant velocity (just for the test with kernel = 3)
                // If we don't do that, the update on densities leads to update on velocities
                // We thus update the momentum with new densities in order to keep velocities (and dv) constants
                AllUc(n,MX1,k,j,i) = rho[n]*vx1[n];
                AllUc(n,MX2,k,j,i) = rho[n]*vx2[n];
                AllUc(n,MX3,k,j,i) = rho[n]*vx3[n];
              }
            }
  );
  data->ConsToPrim();
}
*/