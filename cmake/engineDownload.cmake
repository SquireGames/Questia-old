include(downloadProject)

# TODO see if the project is already available on the computer

set(PROJ_NAME "qeng")

downloadProject(
    PROJECT ${PROJ_NAME}
    REPOSITORY https://github.com/SquireGames/Questia-Engine.git
    COMMIT de885c2
    VERSION_MAJOR 0
    VERSION_MINOR 0
)

include_directories("${${PROJ_NAME}_SOURCE_DIR}/include")
