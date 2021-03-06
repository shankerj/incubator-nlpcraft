/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.nlpcraft.common.extcfg

import java.io._
import java.net.URL
import java.nio.file.Files
import java.util.concurrent.ConcurrentHashMap

import io.opencensus.trace.Span
import org.apache.commons.codec.digest.DigestUtils
import org.apache.commons.io.IOUtils
import org.apache.nlpcraft.common.config.NCConfigurable
import org.apache.nlpcraft.common.extcfg.NCExternalConfigType._
import org.apache.nlpcraft.common.{NCE, NCService, U}
import resource.managed
import scala.collection.JavaConverters._
import scala.io.Source

/**
  * External configuration manager.
  */
object NCExternalConfigManager extends NCService {
    private final val DFLT_DIR = ".nlpcraft/extcfg"
    private final val MD5_FILE = "md5.txt"

    private final val FILES =
        Map(
            GEO → Set(
                "cc_by40_geo_config.zip"
            ),
            SPELL → Set(
                "cc_by40_spell_config.zip"
            ),
            OPENNLP → Set(
                "en-pos-maxent.bin",
                "en-ner-location.bin",
                "en-ner-date.bin",
                "en-token.bin",
                "en-lemmatizer.dict",
                "en-ner-percentage.bin",
                "en-ner-person.bin",
                "en-ner-money.bin",
                "en-ner-time.bin",
                "en-ner-organization.bin"
            )
        )

    private object Config extends NCConfigurable {
        val url: String = getString("nlpcraft.extConfig.extUrl")
        val checkMd5: Boolean = getBool("nlpcraft.extConfig.checkMd5")
        val dir: File = new File(getStringOpt("nlpcraft.extConfig.locDir").getOrElse(s"${U.homeFileName(DFLT_DIR)}"))

        @throws[NCE]
        def check(): Unit = checkAndPrepareDir(Config.dir)
    }

    Config.check()

    private case class Download(fileName: String, typ: NCResourceType) {
        val destDir: File = new File(Config.dir, type2String(typ))
        val file: File = new File(destDir, fileName)
        val isZip: Boolean = {
            val lc = file.getName.toLowerCase

            lc.endsWith(".gz") || lc.endsWith(".zip")
        }
    }

    case class FileHolder(name: String, typ: NCResourceType) {
        val dir = new File(Config.dir, type2String(typ))

        checkAndPrepareDir(dir)

        val file: File = new File(dir, name)
    }

    private object Md5 {
        case class Key(typ: NCResourceType, resource: String)

        private lazy val m: Map[Key, String] = {
            val url = s"${Config.url}/$MD5_FILE"

            try
                managed(Source.fromURL(url)) acquireAndGet { src ⇒
                    src.getLines().map(_.trim()).filter(s ⇒ !s.isEmpty && !s.startsWith("#")).map(p ⇒ {
                        def splitPair(s: String, sep: String): (String, String) = {
                            val seq = s.split(sep).map(_.trim)

                            if (seq.length != 2 || seq.exists(_.isEmpty))
                                throw new NCE(s"Unexpected '$url' file line format: '$p'")

                            (seq(0), seq(1))
                        }

                        val (resPath, md5) = splitPair(p, " ")
                        val (t, res) = splitPair(resPath, "/")

                        Key(string2Type(t), res) → md5
                    }).toList.toMap
                }
            catch {
                case e: IOException ⇒ throw new NCE(s"Failed to read: '$url'", e)
            }
        }

        /**
          *
          * @param f
          * @param typ
          */
        @throws[NCE]
        def isValid(f: File, typ: NCResourceType): Boolean = {
            val v1 = m.getOrElse(Key(typ, f.getName), throw new NCE(s"MD5 data not found for: '${f.getAbsolutePath}'"))

            val v2 =
                try
                    managed(Files.newInputStream(f.toPath)) acquireAndGet { in ⇒ DigestUtils.md5Hex(in) }
                catch {
                    case e: IOException ⇒ throw new NCE(s"Failed to get MD5 for: '${f.getAbsolutePath}'", e)
                }

            v1 == v2
        }
    }

    /**
      * Starts this service.
      *
      * @param parent Optional parent span.
      */
    override def start(parent: Span): NCService = startScopedSpan("start", parent) { _ ⇒
        ackStarting()

        require(NCExternalConfigType.values.forall(FILES.contains))

        val m = new ConcurrentHashMap[NCResourceType, File]

        U.executeParallel(
            NCExternalConfigType.values.flatMap(t ⇒ FILES(t).map(FileHolder(_, t))).toSeq.map(f ⇒ () ⇒ processFile(f, m)): _*
        )

        val downTypes = m.asScala

        if (downTypes.nonEmpty) {
            U.executeParallel(
                downTypes.values.toSeq.map(d ⇒ () ⇒ clearDir(d)): _*
            )
            U.executeParallel(
                downTypes.keys.toSeq.flatMap(t ⇒ FILES(t).toSeq.map(f ⇒ Download(f, t))).map(d ⇒ () ⇒ download(d)): _*
            )
        }

        ackStarted()
    }

    /**
     * Stops this service.
     *
     * @param parent Optional parent span.
     */
    override def stop(parent: Span): Unit = startScopedSpan("stop", parent) { _ ⇒
        ackStopping()
        ackStopped()
    }

    /**
      *
      * @param typ
      * @param res
      * @param parent Parent tracing span.
      */
    @throws[NCE]
    def getContent(typ: NCResourceType, res: String, parent: Span = null): String =
        startScopedSpan("getContent", parent, "res" → res) { _ ⇒
            mkString(U.readFile(mkExtFile(typ, res), "UTF-8"))
        }

    /**
      *
      * @param typ
      * @param res
      * @param parent Parent tracing span.
      */
    @throws[NCE]
    def getStream(typ: NCResourceType, res: String, parent: Span = null): InputStream =
        startScopedSpan("getStream", parent, "res" → res) { _ ⇒
            new BufferedInputStream(new FileInputStream(mkExtFile(typ, res)))
        }

    /**
      * The external resources have higher priority.
      *
      * @param resDir
      * @param resFilter
      * @param parent Parent tracing span.
      */
    @throws[NCE]
    def getDirContent(
        typ: NCResourceType, resDir: String, resFilter: String ⇒ Boolean, parent: Span = null
    ): Stream[NCExternalConfigHolder] =
        startScopedSpan("getDirContent", parent, "resDir" → resDir) { _ ⇒
            val resDirPath = getResourcePath(typ, resDir)

            val d = new File(Config.dir, resDirPath)

            if (!d.exists || !d.isDirectory)
                throw new NCE(s"'${d.getAbsolutePath}' is not a valid folder.")

            val files =
                d.listFiles(new FileFilter { override def accept(f: File): Boolean = f.isFile && resFilter(f.getName) })

            if (files != null)
                files.toStream.map(f ⇒ NCExternalConfigHolder(typ, f.getName, mkString(U.readFile(f, "UTF-8"))))
            else
                Stream.empty
        }

    /**
      *
      * @param h
      * @param m
      */
    @throws[NCE]
    private def processFile(h: FileHolder, m: ConcurrentHashMap[NCResourceType, File]): Unit =
        if (h.file.exists()) {
            if (h.file.isDirectory)
                throw new NCE(s"Unexpected folder (expecting a file): ${h.file.getAbsolutePath}")

            if (h.file.length() == 0 || Config.checkMd5 && !Md5.isValid(h.file, h.typ)) {
                logger.warn(
                    s"File '${h.file.getAbsolutePath}' appears to be corrupted. " +
                        s"All related files will be deleted and downloaded again."
                )

                m.put(h.typ, h.dir)
            }
        }
        else
            m.put(h.typ, h.dir)

    /**
      *
      * @param d
      */
    @throws[NCE]
    private def download(d: Download): Unit = {
        val filePath = d.file.getAbsolutePath
        val url = s"${Config.url}/${type2String(d.typ)}/${d.file.getName}"

        try
            managed(new BufferedInputStream(new URL(url).openStream())) acquireAndGet { src ⇒
                managed(new FileOutputStream(d.file)) acquireAndGet { dest ⇒
                    IOUtils.copy(src, dest)
                }

                logger.info(s"External config downloaded [url='$url', file='$filePath']")
            }
        catch {
            case e: IOException ⇒ throw new NCE(s"Failed to download external config [url='$url', file='$filePath']", e)
        }

        def safeDelete(): Unit =
            if (!d.file.delete())
                logger.warn(s"Couldn't delete file: '$filePath'")

        if (Config.checkMd5 && !Md5.isValid(d.file, d.typ)) {
            safeDelete()

            throw new NCE(s"Unexpected md5 sum for downloaded file: '$filePath'")
        }

        if (d.isZip) {
            val destDirPath = d.destDir.getAbsolutePath

            try {
                U.unzip(filePath, destDirPath)

                logger.trace(s"File unzipped [file='$filePath', dest='$destDirPath']")
            }
            catch {
                case e: NCE ⇒
                    safeDelete()

                    throw e
            }
        }
    }

    /**
      *
      * @param typ
      */
    private def type2String(typ: NCResourceType): String = typ.toString.toLowerCase

    /**
      *
      * @param s
      */
    @throws[NCE]
    private def string2Type(s: String) =
        try
            NCExternalConfigType.withName(s.toUpperCase)
        catch {
            case e: IllegalArgumentException ⇒ throw new NCE(s"Invalid type: '$s'", e)
        }

    /**
      *
      * @param res
      */
    private def mkString(res: Seq[String]): String = res.mkString("\n")

    /**
      *
      * @param d
      */
    @throws[NCE]
    private def checkAndPrepareDir(d: File): Unit =
        if (d.exists()) {
            if (!d.isDirectory)
                throw new NCE(s"'${d.getAbsolutePath}' is not a valid folder.")
        }
        else {
            if (!d.mkdirs())
                throw new NCE(s"'${d.getAbsolutePath}' folder cannot be created.")
        }

    /**
      *
      * @param typ
      * @param res
      */
    private def getResourcePath(typ: NCResourceType, res: String): String = s"${type2String(typ)}/$res"

    /**
      *
      * @param typ
      * @param res
      */
    private def mkExtFile(typ: NCResourceType, res: String): File = new File(Config.dir, getResourcePath(typ, res))

    /**
      *
      * @param d
      */
    @throws[NCE]
    private def clearDir(d: File): Unit = {
        val path = d.getAbsolutePath

        U.clearFolder(path)

        logger.debug(s"Folder cleared: '$path'")
    }
}