use oci_client::secrets::RegistryAuth;
use oci_client::Reference;

/// Resolve registry credentials from a dockerconfig JSON blob.
///
/// If `dockerconfig` is `None`, returns `RegistryAuth::Anonymous`.
pub fn resolve_auth_for_reference(
    dockerconfig: Option<&[u8]>,
    reference: &Reference,
) -> RegistryAuth {
    let config_bytes = match dockerconfig {
        Some(b) => b,
        None => return RegistryAuth::Anonymous,
    };

    let registry = reference.resolve_registry();

    match docker_credential::get_credential_from_reader(config_bytes, registry) {
        Ok(docker_credential::DockerCredential::UsernamePassword(user, pass)) => {
            RegistryAuth::Basic(user, pass)
        }
        Ok(docker_credential::DockerCredential::IdentityToken(_)) => RegistryAuth::Anonymous,
        Err(_) => RegistryAuth::Anonymous,
    }
}
