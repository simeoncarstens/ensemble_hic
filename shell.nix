{ pkgs ? import (fetchTarball https://github.com/NixOS/nixpkgs/archive/34c7eb7545d155cc5b6f499b23a7cb1c96ab4d59.tar.gz) {} }:
with pkgs;
with pkgs.python27Packages;
let 
  csb = buildPythonPackage rec {
    pname = "csb";
    version = "1.2.5";
    src = fetchPypi {
      inherit pname version;
      sha256 = "5acdb655fa290b8b6f0f09faf15bdcefb1795487489c51eba4f57075b92f1a15";
      extension = "zip";
    };
    buildInputs = [ numpy matplotlib scipy ];
    propagatedBuildInputs = [ scipy numpy matplotlib ];
    doCheck = false;
  };

  binf = buildPythonPackage rec {
    pname = "binf";
    version = "0.1.0";
    src = fetchGit {
      url = "git@github.com:simeoncarstens/binf.git";
      rev = "7cc7910895609ee529e48036187843b931b8f81f";
    };
    buildInputs = [ numpy csb scipy matplotlib ];
  };

  rexfw = buildPythonPackage rec {
    pname = "rexfw";
    version = "0.1.0";
    src = fetchGit {
      url = "git@github.com:simeoncarstens/rexfw.git";
      rev = "9e3949bb7360a3fb143ae22c972635fe03d6c12f";
    };
    doCheck = false;
    buildInputs = [ numpy mpi4py openmpi ];
  };

  ensemble_hic = buildPythonPackage {
    pname = "ensemble_hic";
    version = "0.1.0";
    src = pkgs.lib.cleanSource ./.;
    buildInputs = [ numpy cython binf csb scipy rexfw ];
    propagatedBuildInputs = [ numpy binf csb rexfw mpi4py openmpi ];
    installCheckPhase = ''
      cd tests && python run_tests.py
    '';
  };
in
  pkgs.mkShell {
    buildInputs = [ ensemble_hic mpi4py ];
  }

