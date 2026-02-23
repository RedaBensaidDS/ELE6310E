# PATH
exe_name="timeloop-model"
install_bin_dir="${TL_INSTALL_PREFIX}/bin"

# 1) Si l'exécutable n'est PAS trouvé dans le PATH actuel…
if ! command -v "$exe_name" >/dev/null 2>&1; then
  # vérifier que l'exécutable existe bien dans le répertoire cible
  if [ -x "$install_bin_dir/$exe_name" ]; then
    # 2) Ajouter install_bin_dir à PATH s'il n'y est pas déjà (évite les doublons)
    case ":$PATH:" in
      *":$install_bin_dir:"*) : ;;  # déjà présent → ne rien faire
      *) PATH="$install_bin_dir:$PATH"; export PATH;
	 echo "PATH set to $PATH";
	 echo "*** Additional step: The update to PATH can be made persistent by adding this command to your shell startup script such as .bashrc.";
	 MSG="export PATH=\"$install_bin_dir:";
	 MSG+='$PATH"';
	 echo $MSG ; echo ;;
    esac
  else
    echo "Error: $install_bin_dir/$exe_name not found or not executable." >&2
  fi
fi

# LD_LIBRARY_PATH
# add /usr/local/lib to LD_LIBRARY_PATH if not already there
curlibdir="/usr/local/lib"
libpath_updated=0
case ":$LD_LIBRARY_PATH:" in
    *":$curlibdir:"*) : ;; # already there
    *) LD_LIBRARY_PATH="$curlibdir:$LD_LIBRARY_PATH"; export LD_LIBRARY_PATH;
       libpath_updated=1 ;;
esac
# add ${TL_INSTALL_PREFIX}/lib to LD_LIBRARY_PATH if not already there
curlibdir="${TL_INSTALL_PREFIX}/lib"
case ":$LD_LIBRARY_PATH:" in
    *":$curlibdir:"*) : ;; # already there
    *) LD_LIBRARY_PATH="$curlibdir:$LD_LIBRARY_PATH"; export LD_LIBRARY_PATH;
       libpath_updated=1 ;;
esac

if [ $libpath_updated -eq 1 ]
then
    echo "LD_LIBRARY_PATH has been set to: ${LD_LIBRARY_PATH}";
    echo "*** Additional step: The update to LD_LIBRARY_PATH can be made persistent by adding this command to your shell startup script such as .bashrc.";
    MSG="export LD_LIBRARY_PATH=\"${MY_LIB_PATHS}";
    MSG+=':${LD_LIBRARY_PATH}"';
    echo $MSG
fi
