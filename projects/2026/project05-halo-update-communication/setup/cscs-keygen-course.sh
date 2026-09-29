#!/bin/bash
#
# Sign an SSH key for the HPC4WC course account on CSCS.
#
# Works for any course participant: the certificate's principal comes from
# whoever logs in through the browser, not from anything set in this script.
# Only ~/.ssh/config needs your own username (see README).
#
# The old SSHService (sshservice.cscs.ch) and its cscs-keygen.sh script were
# retired in Q2 2026. Key signing now goes through the `cscs-key` CLI, which
# authenticates via OIDC in a browser instead of username/password/OTP.
#
# Docs: https://docs.cscs.ch/access/ssh/
# Tool: https://github.com/eth-cscs/cscs-key

set -euo pipefail

KEY=~/.ssh/cscs-key-course
CSCS_KEY_VERSION=v1.1.0

# Install the cscs-key CLI if it is not on PATH yet.
if ! command -v cscs-key >/dev/null 2>&1; then
    # Pick the release asset matching this machine.
    case "$(uname -s)/$(uname -m)" in
        Linux/x86_64)          target=x86_64-unknown-linux-musl ;;
        Linux/aarch64|Linux/arm64) target=aarch64-unknown-linux-musl ;;
        Darwin/arm64)          target=aarch64-apple-darwin ;;
        Darwin/x86_64)         target=x86_64-apple-darwin ;;
        *)
            echo "No cscs-key build for $(uname -s)/$(uname -m)." >&2
            echo "See https://github.com/eth-cscs/cscs-key/releases" >&2
            exit 1
            ;;
    esac

    echo "cscs-key not found, installing ${CSCS_KEY_VERSION} (${target}) into ~/.local/bin ..."
    mkdir -p "$HOME/.local/bin"
    tarball="cscs-key-${CSCS_KEY_VERSION}-${target}.tar.gz"
    tmpdir=$(mktemp -d)
    trap 'rm -rf "${tmpdir}"' EXIT
    curl -fsSL -o "${tmpdir}/${tarball}" \
        "https://github.com/eth-cscs/cscs-key/releases/download/${CSCS_KEY_VERSION}/${tarball}"
    tar -xzf "${tmpdir}/${tarball}" -C "$HOME/.local/bin"
    export PATH="$HOME/.local/bin:$PATH"

    # The install dir is useless to later shells if it is not on PATH.
    case ":${PATH}:" in
        *":${HOME}/.local/bin:"*) ;;
        *) echo "NOTE: add ~/.local/bin to your PATH to use 'cscs-key' directly." >&2 ;;
    esac
fi

# Generate the key pair once; afterwards we only re-sign it.
if [ ! -f "${KEY}" ]; then
    echo "No key at ${KEY}, generating a new Ed25519 pair ..."
    ssh-keygen -t ed25519 -f "${KEY}" -N "" -C "cscs-hpc4wc-course"
fi

# Sign the public key -> writes ${KEY}-cert.pub, valid for 1 day.
# Add --headless to use the device-authorization flow (prints a code to enter
# on another machine) instead of opening a local browser.
cscs-key sign -f "${KEY}" -d 1d "$@"

echo
echo "Done. Certificate written to ${KEY}-cert.pub"

# The cert's principal is the CSCS account it authenticates as. Read it back so
# the config below is correct without anyone editing a username by hand.
USER_NAME=$(ssh-keygen -L -f "${KEY}-cert.pub" 2>/dev/null \
            | awk '/Principals:/{getline; print $1; exit}')

# santis/clariden have private addresses and are only reachable by jumping
# through ela, so these host entries are required, not just convenient.
if [ -n "${USER_NAME}" ] && ! grep -qE '^Host[[:space:]].*(^|[[:space:]])santis-course([[:space:]]|$)' ~/.ssh/config 2>/dev/null; then
    cat <<EOF

Your ~/.ssh/config has no 'santis-course' entry yet. Append this:

Host ela-course
    HostName ela.cscs.ch
    User ${USER_NAME}
    IdentityFile ${KEY}
    CertificateFile ${KEY}-cert.pub
    IdentitiesOnly yes

Host santis-course
    HostName santis.cscs.ch
    User ${USER_NAME}
    ProxyJump ela-course
    IdentityFile ${KEY}
    CertificateFile ${KEY}-cert.pub
    IdentitiesOnly yes

Host clariden-course
    HostName clariden.cscs.ch
    User ${USER_NAME}
    ProxyJump ela-course
    IdentityFile ${KEY}
    CertificateFile ${KEY}-cert.pub
    IdentitiesOnly yes
EOF
fi

cat <<EOF

Connect with:
  ssh santis-course
  ssh clariden-course
  ssh ela-course

Check validity / list certs:
  cscs-key list

The certificate is valid for 1 day - re-run this script when it expires.
EOF
