pipeline {
    agent {
        docker {
            image 'python:3.11'
        }
    }

    stages {
        stage('Install Dependencies') {
            steps {
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Train Model') {
            steps {
                sh 'python src/train.py'
            }
        }

        stage('Evaluate Model') {
            steps {
                sh 'python src/evaluate.py'
            }
        }

        stage('Show Metrics') {
            steps {
                sh 'cat results/metrics.json'
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'models/*.joblib, results/*', fingerprint: true
            echo 'Artifacts archived.'
        }
        success {
            echo 'Pipeline finished successfully.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
