"""TLS trust.

Networks that inspect TLS (corporate proxies, VPNs, some AV) re-sign HTTPS with a
certificate that Python's bundled ``certifi`` roots do not know, so Hugging Face
downloads fail with ``CERTIFICATE_VERIFY_FAILED: unable to get local issuer
certificate`` even though the browser and the OS trust the connection.

``truststore`` makes Python verify against the operating system's trust store,
which does know that certificate. It is optional: without it, nothing changes.
"""

import logging

logger = logging.getLogger(__name__)

_INJECTED = False


def use_system_certs() -> bool:
    """Verify TLS against the OS trust store. Returns True if that took effect."""
    global _INJECTED
    if _INJECTED:
        return True
    try:
        import truststore
    except ImportError:
        logger.debug("truststore not installed; using the bundled certifi roots")
        return False

    truststore.inject_into_ssl()
    _INJECTED = True
    logger.debug("TLS verification now uses the OS trust store")
    return True
