include("/home/samaria/github/Riemann-Zeta-Tracer/v4.0/client/.qt/QtDeploySupport.cmake")
include("${CMAKE_CURRENT_LIST_DIR}/securesocketclient-plugins.cmake" OPTIONAL)
set(__QT_DEPLOY_I18N_CATALOGS "qtbase")

qt6_deploy_runtime_dependencies(
    EXECUTABLE "/home/samaria/github/Riemann-Zeta-Tracer/v4.0/client/securesocketclient"
    GENERATE_QT_CONF
)
