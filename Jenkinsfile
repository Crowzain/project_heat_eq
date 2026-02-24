pipeline {
    agent any
    
    
    stages {
        stage('Clone') {
            steps {
                echo 'Cloning...'
                git url:'https://github.com/Crowzain/project_heat_eq.git', branch: 'main'
                echo 'Cloned'
            }
        }
        stage('Compose images') {
            steps {
                echo 'Composing...'
                sh 'docker compose up -d --build'
                echo 'Composed'
                sh 'docker compose --profile project_heat_eq down -v'
            }
        }
    }
    post {
        success {
            echo 'Pipeline finished successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
        always {
            echo 'End of pipeline.'
        }
    }
}