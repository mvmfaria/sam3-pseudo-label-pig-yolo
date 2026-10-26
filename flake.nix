{
  description = "Ambiente de desenvolvimento para o repositorio";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    llm-agents.url = "github:numtide/llm-agents.nix";
  };

  outputs = { self, nixpkgs, llm-agents, ... }: let
    system = "x86_64-linux";
    # Permite TODOS os pacotes unfree
    pkgs = import nixpkgs {
      inherit system;
      config.allowUnfree = true;
    };
    agy = llm-agents.packages.${system}.antigravity-cli;
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = with pkgs; [
        agy
        git
        python312
        vscode
        uv
        tmux
      ];
      shellHook = ''
        export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc.lib
          pkgs.libGL
          pkgs.glib
          pkgs.libxcb
          pkgs.libX11
          pkgs.libXext
          pkgs.libXrender
          pkgs.zlib
        ]}:/run/opengl-driver/lib:$LD_LIBRARY_PATH"
        echo "Ambiente Nix ativo com Google Antigravity!"
      '';
    };
  };
}